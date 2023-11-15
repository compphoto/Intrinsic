import torch
from altered_midas.midas_net import MidasNet
from altered_midas.midas_net_custom import MidasNet_small

def load_models(ord_path, mrg_path, device='cuda'):
    """Load the ordinal network and the intrinsic decomposition network
       into a dictionary that can be used to run our pipeline

    params:
        ord_path (str): the path to the weights file for the ord model
        mrg_path (str): the path to the weights file for the mrg model
        device (str) optional: the device to run the model on (default "cuda")

    returns:
        models (dict): a dict with the following structure: {
            "ordinal_model": altered_midas.midas_net.MidasNet,
            "real_model": altered_midas.midas_net_custom.MidasNet_small}
    """
    models = {}

    ord_model = MidasNet()
    ord_model.load_state_dict(torch.load(ord_path))
    ord_model.eval()
    ord_model = ord_model.to(device)

    mrg_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    mrg_model.load_state_dict(torch.load(mrg_path))
    mrg_model.eval()
    mrg_model = mrg_model.to(device)

    models['ordinal_model'] = ord_model
    models['real_model'] = mrg_model

    return models

