import torch
from altered_midas.midas_net import MidasNet
from altered_midas.midas_net_custom import MidasNet_small

def load_models(ord_path, iid_path, device='cuda'):
    """Load the ordinal network and the intrinsic decomposition network
       into a dictionary that can be used to run our pipeline

    params:
        ord_path (str): the path to the weights file for the ordinal model
        iid_path (str): the path to the weights file for the intrinsic decomposition model
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

    iid_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    iid_model.load_state_dict(torch.load(iid_path))
    iid_model.eval()
    iid_model = iid_model.to(device)

    models['ordinal_model'] = ord_model
    models['real_model'] = iid_model

    return models

