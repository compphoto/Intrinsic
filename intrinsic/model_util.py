import torch
from altered_midas.midas_net import MidasNet
from altered_midas.midas_net_custom import MidasNet_small

def load_models(path, device='cuda'):
    """Load the ordinal network and the intrinsic decomposition network
       into a dictionary that can be used to run our pipeline

    params:
        path (str or list): the path to the combined weights file, or to each individual weights file (ordinal first, then iid)
        device (str) optional: the device to run the model on (default "cuda")

    returns:
        models (dict): a dict with the following structure: {
            "ordinal_model": altered_midas.midas_net.MidasNet,
            "real_model": altered_midas.midas_net_custom.MidasNet_small}
    """
    models = {}

    if isinstance(path, list):
        ord_state_dict = torch.load(path[0])
        iid_state_dict = torch.load(path[1])
    else:
        if path == 'paper_weights':
            combined_dict = torch.hub.load_state_dict_from_url('https://github.com/compphoto/Intrinsic/releases/download/v1.0/final_weights.pt', map_location=device, progress=True)
        elif path == 'rendered_only':
            combined_dict = torch.hub.load_state_dict_from_url('https://github.com/compphoto/Intrinsic/releases/download/v1.0/rendered_only_weights.pt', map_location=device, progress=True)
        else:
            combined_dict = torch.load(path)

        ord_state_dict = combined_dict['ord_state_dict']
        iid_state_dict = combined_dict['iid_state_dict']

    ord_model = MidasNet()
    ord_model.load_state_dict(ord_state_dict)
    ord_model.eval()
    ord_model = ord_model.to(device)

    iid_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    iid_model.load_state_dict(iid_state_dict)
    iid_model.eval()
    iid_model = iid_model.to(device)

    models['ordinal_model'] = ord_model
    models['real_model'] = iid_model

    return models

