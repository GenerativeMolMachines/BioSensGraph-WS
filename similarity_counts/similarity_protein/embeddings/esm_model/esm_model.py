import esm

def load_esm_model(model_name: str, device: str):
    if model_name == "esm_2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        raise ValueError(f"Model {model_name} not found")

    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    return model, alphabet, batch_converter