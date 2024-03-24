class DMM_BoundedRational:
    def __init__(self, START_STATE):
        self.state = START_STATE


class DMM_Anchored:
    def __init__(self, START_STATE):
        self.state = START_STATE


class DMM_CPT:
    def __init__(self, START_STATE):
        self.state = START_STATE
        self.behavior = 'seeking'
        #self.behavior = 'averse'

class DMM_Confidence:
    def __init__(self, START_STATE):
        self.state = START_STATE
        self.behavior = 'overconfident'
        #self.behavior = 'underconfident'

