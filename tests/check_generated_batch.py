"""Run separately against a frozen generated bundle, on the remote GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--bundle",required=True)
parser.add_argument("--model",required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(args.bundle).resolve()))
import torch
from transformers import AutoModelForCausalLM
from forge.engine import QwenEngine

torch.set_num_threads(2)
candidate=QwenEngine(args.model)
reference=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,local_files_only=True,attn_implementation="eager").to("cuda").eval()
histories=[[101+i for i in range(n)] for n in (3,17,31)]
maximum=0.0
with torch.inference_mode():
    states=[candidate.prefill(x) for x in histories]
    original=states[:]
    for step in range(4):
        tokens=[201+step,301+step,401+step]
        states=candidate.batch_step(tokens,states)
        for index,token in enumerate(tokens):
            histories[index].append(token)
            expected=reference(torch.tensor([histories[index]],device="cuda")).logits[:,-1,:]
            actual=states[index].last_logits
            maximum=max(maximum,float((actual-expected).abs().max()))
            torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-4)
            assert torch.equal(actual.argmax(-1),expected.argmax(-1))
            assert states[index].length==len(histories[index])
    assert [s.length for s in original]==[3,17,31]
print(json.dumps({"status":"pass","prefix_lengths":[3,17,31],"steps":4,"full_vocab_comparisons":12,
                  "max_abs_error":maximum,"engine_sha256":hashlib.sha256((Path(args.bundle)/'forge/engine.py').read_bytes()).hexdigest()}))
