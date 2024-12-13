# simple-loadbalancer

## Usage

1. Install dependencies
2. Run `python load_balancer.py`
3. Make sure the endpoints in `endpoints_config.yaml` start with `http://`
4. Make sure the endpoints are running and accessible

## Example
start a vllm server on platform  onthingai.com
then get the endpoint url

then edit the endpoints_config.yaml
```
Qwen/Qwen2.5-7B-Instruct: 
  - http://your-endpoint-url-here
```

then run the load_balancer.py

