"""
May 2025

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""


def update_state_dict(old_state_dict, new_state_dict, keypair):
    """
    State dict update. Keys from an old model are replaced with those in the new model.

    Parameters
    __________
    old_state_dict : dict
        The checkpoint containing the saved model state dict (with old keys).
    new_state_dict : dict
        The state dict of the new model (initial state dict to be updated).
    keypair : list of tuple
        A list of tuples, where each tuple contains (old_key_part, new_key_part) to replace in the key.

    Returns
    _______
    new_state_dict : dict
        The updated state dict for the new model with the old keys replaced.
    """

    for old_key, value in old_state_dict.items():
        new_key = old_key

        for old_key_part, new_key_part in keypair:
            if old_key_part in new_key:
                new_key = new_key.replace(old_key_part, new_key_part)

        if new_key in new_state_dict:
            new_state_dict[new_key] = value
        else:
            print(f"Warning: Key '{new_key}' not found in the new model.")

    return new_state_dict
