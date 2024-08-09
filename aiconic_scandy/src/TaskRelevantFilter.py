import torch

class TaskRelevantState:

    def __init__(self):
        self.importance_map = torch.zeros(200,200)

        self.current_time = 0

    def init_state(self, importance_map):
        self.importance_map = importance_map
        self.current_time = 0

    def update_state(self, importance_map):
        self.importance_map = importance_map
        self.current_time = self.current_time + 1

    def create_visualization(self, image_size):
        return self.importance_map.cpu().numpy()

class TaskRelevantFilter:

    def __init__(self, config):
        pass

    def predict(self, state, gaze_state, dt):
        impotance_map = state.importance_map
        return impotance_map

    def correct(self, state, gaze_state, measurement, meas_time):
        dt = meas_time - state.current_time
        impotance_map = self.predict(state, gaze_state, dt)
        impotance_map = measurement / torch.max(measurement)    # normalized!
        state.update_state(impotance_map)