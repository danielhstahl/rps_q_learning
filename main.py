from typing import Dict, List
import numpy as np
import itertools
def get_state_index(outcomes:List[str], state:Dict[str, int])->int:
    if len(outcomes)==0:
        return 0
    last_outcome=outcomes[-1]
    if len(outcomes)==1:
        return state[last_outcome+"trend_w"]
    else:
        if outcomes[-2]=="w":
            return state[last_outcome+"trend_w"]
        elif outcomes[-2]=="l":
            return state[last_outcome+"trend_l"]
        else:
            return state[last_outcome+"no_trend"]


## learn throughout the game, improving based off opponents strategy
def update_q(possible_actions: List[int], q_state:np.array, action_t:str, current_state_index:int, next_state_index:int, reward:float, learning_rate:float, gamma:float):
    action_index=possible_actions.index(action_t)
    q_state[current_state_index, action_index] = (1-learning_rate) * q_state[current_state_index, action_index] +learning_rate*(reward + gamma*max(q_state[next_state_index,:]))


def create_state_space()->List[List[str]]:
    return {elem1+elem2: index for index, [elem1, elem2] in enumerate(itertools.product(["w", "l", "t"], ["trend_w", "trend_l", "no_trend"]))}

def select_action(possible_actions: List[str], q_state:np.array, explore_proba:float, current_state_index:int)->str:
    if np.random.uniform(0,1) < explore_proba:
        action = np.random.choice(possible_actions)
    else:
        action = possible_actions[np.argmax(q_state[current_state_index,:])]
    return action

REWARD={
    "rr":"t",
    "rp":"l",
    "rs":"w",
    "pr":"w",
    "pp":"t",
    "ps":"l",
    "sr":"l",
    "sp":"w",
    "ss":"t"
}
def get_outcome(my_action:str, opponent_action:str)->str:
    return REWARD[my_action+opponent_action]

def get_reward(outcome:str)->float:
    if outcome=="w":
        return 1
    elif outcome=="l":
        return -1
    else:
        return 0.1

if __name__=="__main__":
    print(create_state_space())
    possible_actions=["r", "p", "s"]
    num_actions=len(possible_actions)
    state_space=create_state_space()
    num_states=len(state_space)
    q_state=np.zeros((num_states, num_actions))
    gamma=0.99
    alpha=.7
    explore_proba=0.2
    

    opponent_strategy=lambda: np.random.choice(possible_actions)

    opponent_action=None
    current_state_index=0 # just to start

    outcomes=[]

    for round in range(1000):
        outcome="w" # this will get changed before being used
        if round==0:
            my_action=np.random.choice(possible_actions)
        else:
            my_action=select_action(possible_actions, q_state, explore_proba, current_state_index) # current state index is from the previous game
        opponent_action=opponent_strategy()
        outcome=get_outcome(my_action, opponent_action)
        outcomes.append(outcome)
        next_state_index=get_state_index(outcomes, state_space) 
        reward=get_reward(outcome)
        update_q(possible_actions, q_state, my_action, current_state_index, next_state_index, reward, alpha, gamma)
        current_state_index=next_state_index
        
    
    print(q_state)
    print(sum([1 if outcome=="l" else 0 for outcome in outcomes]))
    print(sum([1 if outcome=="t" else 0 for outcome in outcomes]))
    print(sum([1 if outcome=="w" else 0 for outcome in outcomes]))