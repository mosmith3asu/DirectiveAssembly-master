


class AssemblySimulation(object):
    def __init__(self):
        self.t = 0
        self.duration = 100
        self.States = None
        self.Actions = None
        self.state = None


        self.init_params = {'t': 0,
                       'state': self.state}

    def initialize(self):
        self.init_params = {'t': 0,
                            'state': self.state}

    def step(self):
    def reset(self):
    def reward(self):

    ############################
    # Getters
    ############################
    def get_admissable_actions(self):


