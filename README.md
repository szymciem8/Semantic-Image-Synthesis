# Semantic-Image-Synthesis

![Made with Python](https://img.shields.io/badge/Python-FFD43B?style=flat&logo=python&logoColor=blue)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat&logo=Jupyter)](https://jupyter.org/try)


## Bob Ross AI Painter

You can interact with the models through a Streamlit application. 

![app](/images/bob-ross-ai-painter-screen-1.png)

The app is built with Docker. Use the command below to start the app locally. 
```
docker-compose -f docker-compose.dev.yml up 
```

## Exmaples of GauGAN

<table>
  <tr>
      <td>Segmentation Map</td>
      <td>Ground Truth</td>
      <td>Generated Image</td>
  </tr>
  <tr>
    <td><img src="images/gaugan_input_mask_0.png" width=256></td>
    <td><img src="images/gaugan_ground_truth_0.png" width=256></td>
    <td><img src="images/gaugan_prediction_0.png" width=256></td>
  </tr>
  <tr>
    <td><img src="images/gaugan_input_mask_1.png" width=256></td>
    <td><img src="images/gaugan_ground_truth_1.png" width=256></td>
    <td><img src="images/gaugan_prediction_1.png" width=256></td>
  </tr>
  <tr>
    <td><img src="images/gaugan_input_mask_2.png" width=256></td>
    <td><img src="images/gaugan_ground_truth_2.png" width=256></td>
    <td><img src="images/gaugan_prediction_2.png" width=256></td>
  </tr>
 </table>

## Exmaples of Pix2Pix

<table>
  <tr>
      <td>Segmentation Map</td>
      <td>Ground Truth</td>
      <td>Generated Image</td>
  </tr>
  <tr>
    <td><img src="images/input_mask_0.png" width=256></td>
    <td><img src="images/ground_truth_0.png" width=256></td>
    <td><img src="images/prediction_0.png" width=256></td>
  </tr>
 </table>
