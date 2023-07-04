import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class PerformanceCurve:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def auc(self):
        return metrics.auc(self.x, self.y)

    def plot(self, title, ylabel=None, xlabel=None, xy_line=False):
        plt.title(title)
        plt.plot(self.x, self.y, label=f'AUC = {self.auc():.3f}')
        plt.legend(loc='lower right')
        if xy_line:
            plt.plot([0, 1], [0, 1], '--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()


class ROC(PerformanceCurve):
    def __init__(self, y_test, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, y_pred[:, 1], pos_label=1)
        super().__init__(fpr, tpr)

    def plot(self, title):
        super().plot(f'ROC - {title}',
                     "True positive rate", "False positive rate", xy_line=True)

    @ classmethod
    def calc_auc(cls, y_test, y_pred):
        return cls(y_test, y_pred).auc()


class PRC(PerformanceCurve):
    def __init__(self, y_test, y_pred):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test, y_pred[:, 1], pos_label=1)
        super().__init__(recall, precision)

    def plot(self, title):
        super().plot(f'Precision-Recall Curve - {title}',
                     "Precision", "Recall")

    @ classmethod
    def calc_auc(cls, y_test, y_pred):
        return cls(y_test, y_pred).auc()


def get_feature_names(pipe, verbose=False):
    """
    Get the column names from the a ColumnTransformer containing transformers & pipelines
    Parameters
    ----------
    verbose : a boolean indicating whether to print summaries. 
        default = False
    Returns
    -------
    a list of the correct feature names
    Note: 
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns, 
    it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns 
    to the dataset that didn't exist before, so there should come last in the Pipeline.
    Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525 
    """

    if verbose:
        print('''\n\n---------\nRunning get_feature_names\n---------\n''')

    column_transformer = pipe[0]
    assert isinstance(column_transformer,
                      ColumnTransformer), "Input isn't a ColumnTransformer"
    check_is_fitted(column_transformer)

    new_feature_names, transformer_list = [], []

    for i, transformer_item in enumerate(column_transformer.transformers_):

        transformer_name, transformer, orig_feature_names = transformer_item
        orig_feature_names = list(orig_feature_names)

        if verbose:
            print('\n\n', i, '. Transformer/Pipeline: ', transformer_name, ',',
                  transformer.__class__.__name__, '\n')
            print('\tn_orig_feature_names:', len(orig_feature_names))

        if transformer == 'drop':

            continue

        if isinstance(transformer, Pipeline):
            # if pipeline, get the last transformer in the Pipeline
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):

            if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:

                names = list(transformer.get_feature_names_out(
                    orig_feature_names))

            else:

                names = list(transformer.get_feature_names_out())

        elif hasattr(transformer, 'indicator_') and transformer.add_indicator:
            # is this transformer one of the imputers & did it call the MissingIndicator?

            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag'
                                  for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer, 'features_'):
            # is this a MissingIndicator class?
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag'
                                  for idx in missing_indicator_indices]

        else:

            names = orig_feature_names

        if verbose:
            print('\tn_new_features:', len(names))
            print('\tnew_features:\n', names)

        new_feature_names.extend(names)
        transformer_list.extend([transformer_name] * len(names))

    # self.transformer_list, self.column_transformer_features = transformer_list, new_feature_names

    return new_feature_names


class PermutationImportance:

    def __init__(self, pipe, X, y, n_repeats: int = 10, scoring=['accurcy']):
        self.pipe = pipe
        self.X = X
        self.result = permutation_importance(
            pipe, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1, scoring=scoring
        )

    # def display(self):
    #     for metric in self.scoring:
    #         print(f"{metric}")
    #         r = self.result[metric]
    #         for i in r.importances_mean.argsort()[::-1]:
    #             if r.importances_mean[i] - 2 * r.importances_std[i] > 0:

    def graph(self, title=None, f_title=None, graph_size: tuple = (5.5, 4.25), scoring=None):

        if scoring is None:
            scoring = self.result.keys()

        for metric in scoring:
            r = self.result[metric]
            sorted_idx = r.importances_mean.argsort()

            if not title:
                title = 'Permutation importance'
                if isinstance(self.pipe, Pipeline):
                    title += f' ({self.pipe[-1].__class__.__name__})'

            if f_title:
                title = f_title(metric)

            print(metric)
            fig, ax = plt.subplots()
            ax.boxplot(
                r.importances[sorted_idx].T, vert=False, labels=self.X.columns[sorted_idx]
            )
            ax.set_title(title)
            fig.set_size_inches(*graph_size)
            fig.tight_layout()
            plt.show()


def pretty_print_params(clf, params=None, delimiter=', '):
    if not params:
        params = clf.get_params()

    return delimiter.join([f'{param}: {clf.get_params()[param]}' for param in params])


# def calc_metrics(y_true, y_pred):

#     tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
#     sens = tp / (tp + fn)
#     spec = tn / (tn + fp)
#     prec = tp / (tp + fp)
#     f1 = tp / (tp + (fp + fn)/2)
#     f1c = 2/(1/sens + 1/prec)

#     return {
#         'sensitivity': sens,
#         'sepcificty': spec,
#         'precision': prec,
#         'recall': sens,
#         'f1': f1
#     }

def calc_specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def plot_calibration_curve(y_true, probas_list, clf_names=None, n_bins=10,
                           title='Calibration plots (Reliability Curves)',
                           ax=None, figsize=None, cmap='nipy_spectral',
                           title_fontsize="large", text_fontsize="medium"):

    y_true = np.asarray(y_true)
    if not isinstance(probas_list, list):
        raise ValueError('`probas_list` does not contain a list.')

    classes = np.unique(y_true)
    if len(classes) > 2:
        raise ValueError('plot_calibration_curve only '
                         'works for binary classification')

    if clf_names is None:
        clf_names = ['Classifier {}'.format(x+1)
                     for x in range(len(probas_list))]

    if len(clf_names) != len(probas_list):
        raise ValueError('Length {} of `clf_names` does not match length {} of'
                         ' `probas_list`'.format(len(clf_names),
                                                 len(probas_list)))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i, probas in enumerate(probas_list):
        probas = np.asarray(probas)
        if probas.ndim > 2:
            raise ValueError('Index {} in probas_list has invalid '
                             'shape {}'.format(i, probas.shape))
        if probas.ndim == 2:
            probas = probas[:, 1]

        if probas.shape != y_true.shape:
            raise ValueError('Index {} in probas_list has invalid '
                             'shape {}'.format(i, probas.shape))

        # do not scale/noramlize
        # probas = (probas - probas.min()) / (probas.max() - probas.min())

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, probas, n_bins=n_bins)

        color = plt.cm.get_cmap(cmap)(float(i) / len(probas_list))

        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                label=clf_names[i], color=color)

    ax.set_title(title, fontsize=title_fontsize)

    ax.set_xlabel('Mean predicted value', fontsize=text_fontsize)
    ax.set_ylabel('Fraction of positives', fontsize=text_fontsize)

    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')

    return ax
