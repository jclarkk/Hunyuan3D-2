# Upscalers Installation

## AuraSR-V2

```
pip install aura-sr
```

## Real-ESRGAN

```
pip install realesrgan
```

## InvSR

[Install xformers first](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers)
```
cd ./InvSR
pip install -e ".[torch]"
pip install -r requirements.txt
```

## Flux

```
pip install optimum diffusers -U
```