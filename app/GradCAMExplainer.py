# from app.OmniXAI.omnixai.data.image import Image
# from app.OmniXAI.omnixai.preprocessing.image import Resize
# from app.OmniXAI.omnixai.explainers.vision.specific.gradcam import GradCAM
from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision.specific.gradcam import GradCAM
from PIL import Image as PilImage
import tensorflow as tf
import numpy as np

#python 3.11.0 ->3.10.2 and numpy 1.23.5 needed

class GradCAMExplainer:

    # The preprocessing function
    def preprocess(images):
        data = []
        for i in range(len(images)):
            im = tf.keras.preprocessing.image.img_to_array(images[i].to_pil())
            data.append(np.expand_dims(im, axis=0))
        data = np.concatenate(data, axis=0)
        return data

    def explain_fn(self, img_path, predicted_class_idx, model, class_list):
        # Set the entire model as non-trainable
        model.trainable = False

        # Load the image
        img = Resize((600, 600)).transform(Image(PilImage.open(img_path).convert('RGB')))        

        # Initialize GradCAM with target layer
        layer = model.get_layer('activation') #'top_activation', 'conv2d', 'batch_normalization'

        # Instantiate the grad cam explainer
        explainer = GradCAM(
            model=model,
            target_layer=layer, 
            preprocess_function=GradCAMExplainer.preprocess
        )

        # Explain the top label
        explanations = explainer.explain(img, y=predicted_class_idx) 

        #plot method modified locally
        #call plot method to get the heatmap and score map
        # heatmap, score = explanations.plot(index=0, class_names=class_list) 

        # heatmap[0].savefig('heatmap.jpeg', dpi=600, bbox_inches='tight')
        # score[0].savefig('score.jpeg', dpi=600, bbox_inches='tight')

        # return heatmap[0], score[0]

        heatmap = explanations.plot(index=0, class_names=class_list) 

        heatmap[0].savefig('app/heatmap.jpeg', dpi=600, bbox_inches='tight')

        return heatmap[0]





        

