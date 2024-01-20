class TrainDNA:
    def __init__(self, point=int, state=int, code=str):
        self.point_ = point
        self.state_ = state
        self.code_ = code

    def print(self):
        print("Point " + self.point_)
        print("State " + self.state_)
        print("Code " + self.code_)