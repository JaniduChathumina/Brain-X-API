import numpy as np
from lime import lime_image
from keras.preprocessing import image
import json
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.measure import find_contours

class LimeExplainer:

    class_list = ['Astrocitoma','Carcinoma','Ependimoma','Ganglioglioma','Germinoma','Glioblastoma','Granuloma',
                  'Meduloblastoma','Meningioma','Neurocitoma','Oligodendroglioma','Papiloma','Schwannoma',
                  'Tuberculoma','_NORMAL']
    
    def explain_fn(self, img_path, model):

        # Function to make predictions
        def predict_fn(image):
            predictions = model.predict(image)
            return predictions # returns the predictions array

        # Open and load the image using PIL
        img = image.load_img(img_path, target_size=(600, 600))
        img = np.expand_dims(img, axis=0)

        explainer = lime_image.LimeImageExplainer()
        number_of_classes = 15

        # Explain the prediction for the image
        explanation = explainer.explain_instance(img[0], predict_fn, top_labels=number_of_classes, num_samples=1000) 

        label = explanation.top_labels[0]
        explanations = explanation.local_exp[label]

        # List to store score value outputs with feature and superpixel ids
        output_data = []

        # Sort the explanations by weight in descending order and get the top 10 superpixels
        top_superpixels = sorted(explanations, key=lambda x: x[1], reverse=True)[:10]

        # Create plots
        fig, ax = plt.subplots(1)

        # Display the image
        img = image.load_img(img_path, target_size=(600, 600))
        plt.imshow(img)

        # For each of the top superpixels
        for i, superpixel in enumerate(top_superpixels):
            # Get the mask for the current superpixel
            mask = explanation.segments == superpixel[0]

            # Find contours in the mask
            contours = find_contours(mask, 0.5)

            for contour in contours:
                # Draw each contour line separately
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5)

            # Calculate the centroid of the contour
            props = regionprops(mask.astype(int))
            centroid = props[0].centroid

            # Add a number to the contour at the centroid position
            ax.text(centroid[1], centroid[0], str(i+1), ha='center', va='center', color='red', fontsize=4)

            # Store printed output in the list
            output_data.append({'Superpixel_ID': str(superpixel[0]), 'Feature': str(i+1), 'Weight': str(superpixel[1])})

        plt.title(f'Prediction label : {LimeExplainer.class_list[label]}')
        # plt.show()

        # Saving the figure with high quality
        fig.savefig('output_lime_image.jpeg', dpi=600, bbox_inches='tight')

        # Return image and JSON object
        return fig, output_data

