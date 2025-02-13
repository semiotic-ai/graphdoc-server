

class MLFlowManager:
    def __init__(self, source_tracking_uri: str, target_tracking_uri: str):
        self.source_tracking_uri = source_tracking_uri
        self.target_tracking_uri = target_tracking_uri

    def copy_experiment(self, experiment_name: str):
        pass
