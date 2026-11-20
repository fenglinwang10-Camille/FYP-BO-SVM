"""Read-only search diagnostics; never evaluates an SVM or reads outer results."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    root = Path(__file__).resolve().parent
    source = root / 'results/13_nested_score_tiebreak/split_42_inner_1729/search_42'
    output = root / 'results/15_saved_search_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    names = ['f_low', 'f_high', 'window_size', 'C', 'gamma']
    search_rows, ranges, predictions, gp_rows = [], [], [], []
    for fold in range(1, 6):
        directory = source / f'outer_fold_{fold}'
        optimizer = joblib.load(directory / 'search_state.joblib')['optimizer']
        histories = {method: pd.read_csv(directory / f'{method}_history.csv')
                     for method in ('bo', 'rs')}
        for method, history in histories.items():
            guided = history.iloc[16:50]
            coordinates = np.asarray(optimizer.space.transform(history[names].values.tolist()))
            tail = coordinates[16:50]
            # RMS distance in normalized coordinates: log scales for C and gamma.
            steps = np.linalg.norm(np.diff(tail, axis=0), axis=1) / np.sqrt(5)
            nearest = [np.linalg.norm(coordinates[:i] - coordinates[i], axis=1).min()
                       / np.sqrt(5) for i in range(16, 50)]
            best_index = int(np.argmax(history.score.to_numpy()))
            best = coordinates[best_index]
            search_rows.append(dict(
                outer_fold=fold, method=method.upper(), best_iteration=best_index + 1,
                best_cv_percent=float(history.score.max() * 100),
                gain_after_16_pp=float((history.score.max()-history.score.iloc[:16].max())*100),
                median_step_rms=float(np.median(steps)),
                median_nearest_previous_rms=float(np.median(nearest)),
                dispersion_rms=float(np.sqrt(np.mean(np.var(tail, axis=0)))),
                selected_boundary_count=int(np.sum((best <= 1e-8) | (best >= 1-1e-8)))))
            for index, name in enumerate(names):
                ranges.append(dict(outer_fold=fold, method=method.upper(), parameter=name,
                                   minimum=float(guided[name].min()), maximum=float(guided[name].max()),
                                   normalized_range=float(np.ptp(tail[:, index]))))
        # These RS observations were never supplied to this final GP.
        rs = histories['rs'].iloc[16:50]
        points = rs[names].to_numpy()
        seen = np.any(np.all(np.isclose(points[:, None, :], np.asarray(optimizer.Xi)[None, :, :],
                                       rtol=1e-10, atol=1e-12), axis=2), axis=1)
        unused = rs.loc[~seen].copy()
        means = -optimizer.models[-1].predict(optimizer.space.transform(unused[names].values.tolist()))
        actual = unused.score.to_numpy()
        baseline = -np.mean(optimizer.yi)
        gp_rows.append(dict(outer_fold=fold, unused_rs_candidates=len(unused),
                            gp_mae_pp=float(np.mean(np.abs(means-actual))*100),
                            constant_training_mean_mae_pp=float(np.mean(np.abs(baseline-actual))*100),
                            spearman=float(spearmanr(actual, means).statistic)))
        unused['outer_fold'] = fold
        unused['gp_prediction_percent'] = means * 100
        unused['actual_cv_percent'] = actual * 100
        predictions.extend(unused[['outer_fold','iteration']+names+
                                  ['gp_prediction_percent','actual_cv_percent']].to_dict('records'))
    for filename, rows in [('search_geometry.csv', search_rows), ('parameter_ranges.csv', ranges),
                           ('gp_unseen_rs_metrics.csv', gp_rows), ('gp_unseen_rs_predictions.csv', predictions)]:
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    (output / 'README.md').write_text(
        'Diagnostics for experiment 13, original 50 evaluations only. No new model search or outer-test analysis.\n\n'
        'Distances use the saved skopt normalized search coordinates, including logarithmic C and gamma. '
        'RMS distances divide Euclidean distance by sqrt(5). Boundary tolerance is 1e-8. '
        'Parameter ranges use evaluations 17-50; selected boundaries refer to the earliest highest-scoring candidate.\n\n'
        'The final GP is assessed on RS candidates not observed by that GP. This is an independent-candidate '
        'check on the same inner folds, not independent-data generalization or a prequential acquisition test. '
        'The constant baseline is the mean of scores supplied to the GP. '
        'Geometry alone cannot diagnose xi or establish convergence.\n', encoding='utf-8')
    print(json.dumps({'geometry': search_rows, 'gp': gp_rows}, indent=2))


if __name__ == '__main__':
    main()
