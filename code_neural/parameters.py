import numpy as np

class parameters:
    def __init__(self):

        # Teacher's args chain
        self.teacher_args_chain = {"ExploB_lmbd": 1.0,
                               "ExploB_max": 1.0,
                               "eta_phi_SelfRS": 1e-3,
                                "eta_phi_sors": 1e-3,
                                "eta_phi_lirpg": 1e-3,
                                "eta_critic": 5e-3,
                                "sors_n_pairs": 10,
                                "clipping_epsilon": 0.1
                               }

        # Learner's args
        self.agent_args = {"eta_actor": 1e-5,
                           "eta_critic": 5e-5
                          }

        # Teaching's args
        self.teaching_args = {
            "N_reinforce": 50001,
            "N_r": 20,
            "N_p": 2,  # 5
            "buffer_size": 10,
            "buffer_size_recent": 5,
            "agent_evaluation_step": 100
        }
    # endde
    #endde
#endcalss