# API

You could launch an API server locally, which you could post web request for Image/Text to 3D, Texturing existing mesh,
and e.t.c.

```bash
python api_server.py --host 0.0.0.0 --port 8080
```

A demo post request for image to 3D without texture.

```bash
img_b64_str=$(base64 -i assets/demo.png)
curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "image": "'"$img_b64_str"'",
         }' \
     -o test2.glb
```
