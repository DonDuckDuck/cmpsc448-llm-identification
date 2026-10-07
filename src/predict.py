"""Local demonstration using a trusted initial or optimized checkpoint."""
import argparse,json
import torch
from train import ROOT,LABELS,TextData,collate
from models import TextCNN,TextLSTM
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='models/arena_cnn_output_s1448.pt')
    p.add_argument('--response',default='');p.add_argument('--prompt',default='')
    args=p.parse_args();saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=saved['config'];mode=saved.get('mode',cfg.get('mode','output'))
    if mode in ['output','combined'] and not args.response:p.error('--response is required')
    if mode in ['input','combined'] and not args.prompt:p.error('--prompt is required')
    labels=saved.get('labels',LABELS)
    if saved.get('dataset')=='arena':
        from arena_data import mask
        from arena_train import Data
        from models_v2 import make_model
        row={'LLM_Input':mask(args.prompt),'LLM_output':mask(args.response),'family':labels[0]}
        data=Data([row],saved['vocab'],mode,cfg);model=make_model(cfg,len(saved['vocab']))
    elif 'architecture' in cfg:
        from prepare_data import mask
        row={'LLM_Input':mask(args.prompt),'LLM_output':mask(args.response),'family':labels[0]}
        from iterate import Data
        from models_v2 import make_model
        data=Data([row],saved['vocab'],mode,cfg);model=make_model(cfg,len(saved['vocab']))
    else:
        from prepare_data import mask
        row={'LLM_Input':mask(args.prompt),'LLM_output':mask(args.response),'family':labels[0]}
        data=TextData([row],saved['vocab'],mode,cfg.get('normalized',False))
        cls=TextCNN if cfg['model']=='cnn' else TextLSTM;model=cls(len(saved['vocab']))
    tokens,lengths,_=collate([data[0]]);model.load_state_dict(saved['state_dict']);model.eval()
    with torch.inference_mode():probs=model(tokens,lengths).softmax(1)[0].tolist()
    scope=('Only the four specific Claude/GPT/Grok/DeepSeek releases in the curated Arena study.' if saved.get('dataset')=='arena' else 'Only Falcon/GPT/Llama releases represented in the historical dataset.')
    print(json.dumps({'predicted_family':labels[max(range(len(labels)),key=lambda i:probs[i])],
                     'uncalibrated_model_probabilities':dict(zip(labels,probs)),
                     'scope':scope+' Not a reliable attribution tool for arbitrary text.'},indent=2))
