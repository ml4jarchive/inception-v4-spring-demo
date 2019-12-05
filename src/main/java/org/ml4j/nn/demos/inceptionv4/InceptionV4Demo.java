package org.ml4j.nn.demos.inceptionv4;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.LogManager;

import javax.imageio.ImageIO;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.demos.inceptionv4.util.BufferedImageFeaturesMapper;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.InceptionV4Labels;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.supervised.FeedForwardNeuralNetworkContextImpl;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class InceptionV4Demo implements CommandLineRunner {

	private static final Logger LOGGER = LoggerFactory.getLogger(InceptionV4Demo.class);

	@Autowired
	private MatrixFactory matrixFactory;

	@Autowired
	private InceptionV4Factory inceptionV4Factory;

	public static void main(String[] args) throws Exception {

		// Quieten Logging
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();

		// Application entry-point
		SpringApplication.run(InceptionV4Demo.class, args);
	}

	@Override
	public void run(String... args) throws Exception {

		// Create a runtime (non-training) context for the Inception V4 Network
		FeedForwardNeuralNetworkContext predictionContext = new FeedForwardNeuralNetworkContextImpl(matrixFactory, false);
		predictionContext.setTrainingMiniBatchSize(2);
		
		// Create the Inception V4 Network, configuring the prediction context
		SupervisedFeedForwardNeuralNetwork inceptionV4Network = inceptionV4Factory.createInceptionV4(predictionContext);
		
		// Create the neuron activations for the input image
		NeuronsActivation imageNeuronsActivation = getImageActivations();

		LOGGER.info("Forward propagating images through Inception V4...");
		
		// Forward propagate the image neurons activation through the Inception V4
		// Network
		ForwardPropagation result = inceptionV4Network.forwardPropagate(imageNeuronsActivation, predictionContext);
		
		LOGGER.info("Obtained predictions");

		// Obtain the predictions for the batch from the result
		Matrix classificationProbablitiesForBatch = result.getOutput().getActivations(matrixFactory);
		
		// Obtain the labels for the InceptionV4 Network
		InceptionV4Labels inceptionV4ClassificationNames = inceptionV4Factory.createInceptionV4Labels();

		for (int column = 0; column < classificationProbablitiesForBatch.getColumns(); column++) {
			Matrix classificationProbablitiesForExample = classificationProbablitiesForBatch.getColumn(column);
			int predictedIndex = classificationProbablitiesForExample.argmax();
			String predictedClassificationName = inceptionV4ClassificationNames.getLabel(predictedIndex);
			LOGGER.info("Predicted : '" + predictedClassificationName + "' with probability " + classificationProbablitiesForExample.get(predictedIndex));
		}
	}
	
	private NeuronsActivation getImageActivations(List<BufferedImage> bufferedImages) throws IOException {
		List<float[]> imagesList = new ArrayList<>();
		BufferedImageFeaturesMapper mapper = new BufferedImageFeaturesMapper(299 , 299);

		for (BufferedImage bufferedImage : bufferedImages){
			imagesList.add(mapper.toFeaturesVector(bufferedImage));
		}
		float[][] images = new float[imagesList.size()][mapper.getFeatureCount()];
		int i = 0;
		for (float[] image : imagesList) {
			images[i++] = image;
		}
		return new NeuronsActivationImpl(matrixFactory.createMatrixFromRows(images).transpose(), 
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}
	
	
	private NeuronsActivation getImageActivations() throws IOException {
		File imagesDirectory = new File(InceptionV4Demo.class.getClassLoader().getResource("test_images").getFile());
		List<BufferedImage> bufferedImages = new ArrayList<>();
		for (File imageFile : imagesDirectory.listFiles(f -> f.getPath().endsWith(".jpg"))) {
			LOGGER.info("Loaded image:" + imageFile.getName());
			BufferedImage bufferedImage = ImageIO.read(imageFile);		
			bufferedImages.add(bufferedImage);
		}
		return getImageActivations(bufferedImages);
	}
}
