

from dataclasses import dataclass
from typing import Tuple

from utils.features import Feature, FeatureList, Ftype


@dataclass
class Experiment:
    name: str
    desc: str
    fts: FeatureList

    def run(self) -> Tuple[str, FeatureList]:
        return (self.name, self.desc, self.fts)


exp_endoscopist_specific = Experiment(
    name='ssl-doc',
    desc='Endoscopist-specific models',
    fts=FeatureList([
        Feature('pt_sex', Ftype.dichotomous),
        Feature('pt_age', Ftype.continuous),
        Feature('pt_diabetic', Ftype.dichotomous),
        Feature('pt_fit', Ftype.dichotomous),

        Feature('risk_avg', Ftype.dichotomous),
        Feature('risk_fhx', Ftype.dichotomous),
        Feature('risk_polyp', Ftype.dichotomous),
        Feature('risk_ca', Ftype.dichotomous),

        Feature('doc_id', Ftype.categorical),
        # Feature('doc_adr', Ftype.continuous),
        Feature('doc_spec', Ftype.dichotomous),

        Feature('proc_year', Ftype.categorical),
        Feature('proc_wt6', Ftype.dichotomous),
        Feature('proc_diff', Ftype.dichotomous),
        Feature('prep_poor', Ftype.dichotomous),

        Feature('path_adenoma', Ftype.dichotomous),
        Feature('path_adv_ad', Ftype.dichotomous),
        Feature('path_hp', Ftype.dichotomous),
        Feature('path_ssl', Ftype.dichotomous, target=True),
        Feature('path_tsa', Ftype.dichotomous),
    ])
)

exp_endoscopist_agnostic = Experiment(
    name='ssl-adr',
    desc='Endoscopist-agnostic models',
    fts=FeatureList([
        Feature('pt_sex', Ftype.dichotomous),
        Feature('pt_age', Ftype.continuous),
        Feature('pt_diabetic', Ftype.dichotomous),
        Feature('pt_fit', Ftype.dichotomous),

        Feature('risk_avg', Ftype.dichotomous),
        Feature('risk_fhx', Ftype.dichotomous),
        Feature('risk_polyp', Ftype.dichotomous),
        Feature('risk_ca', Ftype.dichotomous),

        # Feature('doc_id', Ftype.categorical),
        Feature('doc_adr', Ftype.continuous),
        Feature('doc_spec', Ftype.dichotomous),

        Feature('proc_year', Ftype.categorical),
        Feature('proc_wt6', Ftype.dichotomous),
        Feature('proc_diff', Ftype.dichotomous),
        Feature('prep_poor', Ftype.dichotomous),

        Feature('path_adenoma', Ftype.dichotomous),
        Feature('path_adv_ad', Ftype.dichotomous),
        Feature('path_hp', Ftype.dichotomous),
        Feature('path_ssl', Ftype.dichotomous, target=True),
        Feature('path_tsa', Ftype.dichotomous),
    ])
)
