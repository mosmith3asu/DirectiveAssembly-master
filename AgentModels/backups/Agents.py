from matplotlib.patches import Rectangle,Circle

class AgentsOBJ:
    def __init__(self,ID,DMM,state,ego_k=1,partner_k=1):
        self.ID = ID
        self.state = state
        self.update_icon()
        # self.x = 0 # x_loc of agent
        # self.y = 0 # y_loc of agent
        # self.r = 0 # rotation of agent manipulator \in [0,3] left/right 90deg rotations
        # self.g = 0 # 0 is not grasping block 1 is grasping block
        # self.state=[self.x,self.y,self.r,self.g]

        """
        self.has_block: 0 if there is not block in agents hand 1 if there is
        self.agent_k: order of recursive reasoning
        self.agent_intent: agents intent or belief about others intentions
        self.agent_DMM: agent's decision-making model (DMM)= rational, noisy rational, or heuristic
        """
        self.ego_k=1
        self.has_block = 0

        self.ego_k = 1
        self.ego_intent = DMM
        self.ego_DMM = DMM

        self.partner_k = 1
        self.partner_intent = None
        self.partner_DMM = None

    def update_icon(self):
        icon_h, icon_w = (1, 1)
        if self.ID == 1:
            self.agent_color = "red"
            self.icon = Rectangle((self.state[0], self.state[1]), icon_w, icon_h,
                                   linewidth=1,edgecolor='k', facecolor=self.agent_color)
        elif self.ID == 2:
            self.agent_color = "blue"
            self.icon = Circle((self.state[0], self.state[1]), icon_w/2,
                               linewidth=1, edgecolor='k', facecolor=self.agent_color)
        self.grip_alpha = [0.5, 1]
