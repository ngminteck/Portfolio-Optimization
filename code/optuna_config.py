MAX_TRIALS = 50

class EarlyStoppingCallback:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_value = None
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        if study.best_trial.value == self.best_value:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.best_value = study.best_trial.value

        if self.no_improvement_count >= self.patience:
            study.stop()