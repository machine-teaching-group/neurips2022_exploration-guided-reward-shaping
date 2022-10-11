import numpy as np

class parameters:
    def __init__(self):

        # Teacher's args chain
        self.teacher_args_chain = {"ExploB_lmbd": 1.0,
                               "ExploB_max": 1.0,
                               "eta_phi_SelfRS": 0.01,
                                "eta_phi_sors": 0.01,
                                "eta_phi_rlirpg": 0.01,
                                "eta_critic": 0.01,
                                "sors_n_pairs": 10,
                                "clipping_epsilon": 0.01,
                                "use_clipping": False
                               }

        # Teacher's args 4room
        self.teacher_args_4room = {"ExploB_lmbd": 1.0,
                               "ExploB_max": 1.0,
                               "eta_phi_SelfRS": 0.01,
                                "eta_phi_sors": 0.01,
                                "eta_phi_rlirpg": 0.01,
                                "eta_critic": 0.01,
                                "sors_n_pairs": 10,
                                "clipping_epsilon": 0.01,
                                "use_clipping": False
                               }

        # Learner's args
        self.agent_args = {"eta_actor": 0.1,
                            "Q_epsilon": 0.05,
                             "Q_alpha": 0.1,
                          }

        # Teaching's args
        self.teaching_args = {
            "N_reinforce": 50001,
            "N_Q_learning": 50001,
            "N_r": 5,
            "N_p": 2,
            "buffer_size": 10,
            "buffer_size_recent": 5,
            "agent_evaluation_step": 100
        }
    #endde
#endcalss