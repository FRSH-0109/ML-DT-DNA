import numpy as np
from enum import Enum

class DnaAttrVal(Enum):
        G = 'G'
        C = 'C'
        T = 'T'
        A = 'A'
        N = 'N'
        S = 'N'

        def __lt__(self, other):
            enum1 = self.value
            enum2 = other.value
            return enum1 < enum2

class TrainDNA:
    def __init__(self, point=int, state=int, code=str):
        self.point_ = int(point)
        self.state_ = int(state)
        self.code_ = code
        self.length_ = len(code)
        self.attributes_ = []
        self.attributes_ = self.createAttributes()

    def print(self):
        print("Point " + str(self.point_))
        print("State " + str(self.state_))
        print("Code " + str(self.code_))

    def createAttributes(self):
        attributes = []
        for dna_char in self.code_:
            attr = ''
            if dna_char == 'A':
                attr =  DnaAttrVal.A
            elif dna_char == 'C':
                attr = DnaAttrVal.C
            elif dna_char == 'G':
                attr = DnaAttrVal.G
            elif dna_char == 'N':
                attr = DnaAttrVal.N # Verify if N is intentional in the data
            elif dna_char == 'S':
                attr = DnaAttrVal.S # Verify if S is intentional in the data
            elif dna_char == 'T':
                attr = DnaAttrVal.T
            attributes.append(attr)
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
