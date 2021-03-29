
# creating a py file for the model, note that the %% command needs to be in the first line
# due to time contraints, using bentoML to simplify process

from bentoml import BentoService, api, env, artifacts
from bentoml.adapters import DataframeInput
from bentoml.service.artifacts.common import PickleArtifact

@artifacts([PickleArtifact('model')])
@env(conda_dependencies=["scikit-learn"])


class TelcoChurnClassifier(BentoService):

    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df)
