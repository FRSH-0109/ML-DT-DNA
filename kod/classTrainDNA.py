class TrainDNA:
    def __init__(self, point=int, state=int, code=str):
        self.point_ = point
        self.state_ = state
        self.code_ = code
        self.length_ = len(code)
        self.attributes_ = []
        self.attributes_ = self.createAttributes()

    def print(self):
        print("Point " + self.point_)
        print("State " + self.state_)
        print("Code " + self.code_)
    
    def createAttributes(self):
        attributes = []
        for dna_char in self.code_:
            attributes.append(dna_char)
        return attributes
    
    def setPoint(self, new_point=int):
        self.point_ = new_point

    def getPoint(self):
        return self.point_
    
    def setState(self, new_state=int):
        self.state_ = new_state

    def getState(self):
        return self.state_
    
    def setCode(self, new_code=str):
        self.code_ = new_code
        self.length_ = len(new_code)
        self.attributes_ = self.createAttributes()

    def getCode(self):
        return self.code_
    
    def getLength(self):
        return self.code_
    
    def getAttributes(self):
        return self.attributes_
