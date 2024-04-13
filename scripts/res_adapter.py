import os
import gradio as gr
import modules.ui
import modules.scripts as scripts
import modules.shared as shared
from modules import script_callbacks, sd_models, extra_networks
from modules import sd_hijack
from modules import devices, lowvram
from modules.scripts import basedir

from safetensors import safe_open


# check compatibility
sdnext = False
try:
    from modules import sd_unet
except Exception as e:
    sd_unet = None
    print("No sd_unet module found. this is SD.Next. ignore.")
    sdnext = True

try:
    send_model_to_cpu = sd_models.send_model_to_cpu
    send_model_to_device = sd_models.send_model_to_device
except Exception as e:
    def send_model_to_cpu(m):
        if getattr(m, "lowvram", False):
            lowvram.send_everything_to_cpu()
        else:
            m.to(devices.cpu)

        devices.torch_gc()


    def send_model_to_device(m):
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.setup_for_low_vram(m, not shared.cmd_opts.lowvram)
        else:
            m.lowvram = False

        if not getattr(m, "lowvram", False):
            m.to(shared.device)

scriptdir = basedir()


def unet_blocks_map(diffusion_model, isxl=False):
    block_map = {}
    block_map['time_embed.'] = diffusion_model.time_embed

    BLOCKLEN = 12 - (0 if not isxl else 3)
    for j in range(BLOCKLEN):
        block_name = f"input_blocks.{j}."
        block_map[block_name] = diffusion_model.input_blocks[j]

    block_map["middle_block."] = diffusion_model.middle_block

    for j in range(BLOCKLEN):
        block_name = f"output_blocks.{j}."
        block_map[block_name] = diffusion_model.output_blocks[j]

    block_map["out."] = diffusion_model.out

    return block_map


def _make_unet_conversion_map_layer(v2=False, sdxl=False):
    """Simplified UNet Conversion map layer"""
    unet_conversion_map_layer = []
    for i in range(4) if not sdxl else range(3):
        # loop over downblocks/upblocks

        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        for j in range(3) if not sdxl else range(4):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    return unet_conversion_map_layer


def make_unet_conversion_map(v2=False, sdxl=False):
    """Simplified UNet conversion map"""
    unet_conversion_map_layer = _make_unet_conversion_map_layer(v2, sdxl)

    unet_conversion_map = []
    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("out_layers.0.", "norm2."),
    ]

    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))

    return unet_conversion_map

# simplified/modified get_unet_conversion_map from kohya_ss script
def get_unet_conversion_map(state_dict_or_keys, v2=False, sdxl=False):
    unet_conversion_map = make_unet_conversion_map(v2, sdxl)
    conversion_map = {hf: sd for sd, hf in unet_conversion_map}

    mapping = {}
    keys = state_dict_or_keys if type(state_dict_or_keys) == list else state_dict_or_keys.keys()

    for key in keys:
        key_fragments = key.split(".")[:-1] # remove weight/bias
        while len(key_fragments) > 0:
            key_prefix = ".".join(key_fragments) + "."
            if key_prefix in conversion_map:
                converted_prefix = conversion_map[key_prefix]
                converted_key = converted_prefix + key[len(key_prefix) :]
                mapping[key] = converted_key
                break
            key_fragments.pop(-1)
        assert len(key_fragments) > 0, f"key {key} not found in conversion map"

    return mapping


def convert_unet_state_dict(unet_state_dict, v2=False, sdxl=False):
    #if 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight' in unet_state_dict:
    #    v2 = state_dict_base['down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight'].shape[1] == 1024
    #if "mid_block.attentions.0.transformer_blocks.4.attn2.to_q.weight" in unet_state_dict:
    #    sdxl = True

    mapping = get_unet_conversion_map(unet_state_dict, v2, sdxl)

    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

    #if v2:
    #    conv_transformer_to_linear(new_state_dict)

    return new_state_dict


class ResAdapterScript(scripts.Script):
    NORM_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
    LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"

    def __init__(self):
        super().__init__()

    def title(self):
        return "Res Adapter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Res Adapter", open=False) as res:
            with gr.Row():
                use_resadapter = gr.Checkbox(label="Enable Res-Adapter", value=False, visible=True, elem_classes=["res_adapter_enabled"])

            with gr.Row():
                res_version = gr.Radio(label="Select Res-Adapter version",
                    choices=[("v1 sd1.5", "v1_sd1.5"), ("v2 sd1.5", "v2_sd1.5"), ("v1 sdxl", "v1_sdxl"), ("v2 sdxl", "v2_sdxl")],
                    value="v2_sd1.5",
                    interactive=True)

        return [use_resadapter, res_version]


    def before_process(self, p, enabled, version, *args_):
        if not enabled:
            if shared.sd_model is not None and getattr(shared.sd_model, "orig_norm_state_dict", None) is not None:
                print("Restore original norm state_dict ...")
                self.apply_norm_state_dict(shared.sd_model, shared.sd_model.orig_norm_state_dict)
                shared.sd_model.fix_resadapter = None
                shared.sd_model.resadapter_version = None

            return

        if shared.sd_model is not None and getattr(shared.sd_model, "fix_resadapter", None) is not None:
            if shared.sd_model.resadapter_version == version:
                return

        norm_state_dict = {}
        print(f"Load Res-Adapter version {version}...")
        norm_path = os.path.join(scriptdir, "models", "res_adapter", f"resadapter_{version}", self.NORM_WEIGHTS_NAME)
        try:
            unet_state_dict = {}
            with safe_open(norm_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    unet_state_dict[k] = f.get_tensor(k)
                # convert hf state_dict to sd state_dict
                norm_state_dict = convert_unet_state_dict(unet_state_dict)
        except:
            print("There is no normalization safetensors, we can only load lora safetensors for resolution interpolation.")

        if len(norm_state_dict) > 0 and shared.sd_model is not None:
            orig_state_dict = self.apply_norm_state_dict(shared.sd_model, norm_state_dict)
            # store original norm_state_dict
            if getattr(shared.sd_model, "orig_norm_state_dict", None) is None:
                shared.sd_model.orig_norm_state_dict = orig_state_dict.copy()
            shared.sd_model.resadapter_version = version


    def apply_norm_state_dict(self, sd_model, norm_state_dict):
        orig_state_dict = {}

        if len(norm_state_dict) > 0 and sd_model is not None:
            norm_changed_blocks = set()
            for k in norm_state_dict.keys():
                tmp = k.split(".")
                if tmp[0] in [ "input_blocks", "output_blocks" ]:
                    block = f"{tmp[0]}.{tmp[1]}."
                    norm_changed_blocks.add(block)

                elif tmp[0] == "middle_block":
                    norm_changed_blocks.add("middle_block.")

            norm_changed_blocks = list(sorted(norm_changed_blocks))

            weight_changed = {}
            if len(norm_changed_blocks) > 0:
                # get changed keys
                for k in norm_state_dict.keys():
                    for s in norm_changed_blocks:
                        if s not in ["cond_stage_model.", "conditioner."]:
                            #ss = f"model.diffusion_model.{s}"
                            ss = f"{s}"
                        else:
                            ss = s
                        if ss in k:
                            weight_changed[s] = weight_changed.get(s, [])
                            weight_changed[s].append(k)
                            break

            # get unet_blocks_map
            unet_map = unet_blocks_map(sd_model.model.diffusion_model, sd_model.is_sdxl)

            # to cpu ram
            if sd_unet is not None:
                sd_unet.apply_unet("None")
            send_model_to_cpu(sd_model)
            sd_hijack.model_hijack.undo_hijack(sd_model)

            # partial update unet blocks state_dict
            unet_updated = 0
            for s in norm_changed_blocks:
                shared.state.textinfo = "Update UNet Blocks..."
                print(" - update UNet block", s)
                unet_dict = unet_map[s].state_dict()
                #prefix = f"model.diffusion_model.{s}" # no need to prepend 'model.diffusion_model.'
                prefix = f"{s}"
                for k in weight_changed[s]:
                    # remove block prefix, 'model.diffusion_model.input_blocks.0.' will be removed
                    key = k[len(prefix):]
                    orig_state_dict[k] = unet_dict[key].clone().detach()
                    unet_dict[key] = norm_state_dict[k]
                unet_map[s].load_state_dict(unet_dict)
                unet_updated += 1

            if unet_updated > 0:
                print(" - \033[92mUNet res_adapter blocks have been successfully updated\033[0m")

                # add fix_resadapter attribute
                sd_model.fix_resadapter = True

            # restore to gpu
            send_model_to_device(sd_model)
            sd_hijack.model_hijack.hijack(sd_model)

            sd_models.model_data.set_sd_model(sd_model)
            if sd_unet is not None:
                sd_unet.apply_unet()

            return orig_state_dict
