package org.ml4j.nn.demos.inceptionv4;

import java.io.File;
import java.nio.file.Path;
import java.util.function.Supplier;
import java.util.logging.LogManager;
import java.util.stream.Stream;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Image;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.datasets.BatchedDataSet;
import org.ml4j.nn.datasets.FeatureExtractionErrorMode;
import org.ml4j.nn.datasets.featureextraction.ImageSupplierFeatureExtractor;
import org.ml4j.nn.datasets.images.DirectoryImagesWithPathsDataSet;
import org.ml4j.nn.datasets.images.LabeledImagesDataSet;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationDataSet;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.InceptionV4Labels;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
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
	private DirectedComponentsContext directedComponentsContext;

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
	
		// Create the data set
		
		// Define images Directory
		Path imagesDirectory = new File(InceptionV4Demo.class.getClassLoader().getResource("test_images").getFile()).toPath();
		
		// Define data set of images from a directory labelled with the path names
		LabeledImagesDataSet<String> labeledImagesDataSet = 
							new DirectoryImagesWithPathsDataSet(imagesDirectory, 
									path -> true, 299, 299)
								.getLabeledImagesDataSet((index, pathName) -> 
									pathName.getParent().getFileName().toString());
			
		// Map to a data set of image batches with more than 2 in each batch
		BatchedDataSet<Supplier<Image>> batchedImageSupplierDataSet = labeledImagesDataSet.getDataSet().toBatchedDataSet(2);
		
		// Map to a data set of NeuronsActivation instances, one for each batch
		NeuronsActivationDataSet neuronsActivationDataSet = batchedImageSupplierDataSet
				.toFloatArrayBatchedDataSet(new ImageSupplierFeatureExtractor(299 * 299 * 3), FeatureExtractionErrorMode.RAISE_EXCEPTION).toNeuronsActivationDataSet(matrixFactory,
						ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);
		
		// Create the prediction context and the neural network
		
		// Create a runtime (non-training) context for the Inception V4 Network
		FeedForwardNeuralNetworkContext predictionContext = new FeedForwardNeuralNetworkContextImpl(
				directedComponentsContext, false);
		
		// Create the Inception V4 Network, configuring the prediction context
		SupervisedFeedForwardNeuralNetwork inceptionV4Network = inceptionV4Factory.createInceptionV4(predictionContext);
			
		// Make the predictions
		
		Stream<ForwardPropagation> forwardPropagations = inceptionV4Network
				.forwardPropagate(neuronsActivationDataSet.stream(), predictionContext);
		
		// Output the predictions
		
		
		// Obtain the labels for the InceptionV4 Network
		InceptionV4Labels inceptionV4ClassificationNames = inceptionV4Factory.createInceptionV4Labels();
	
		// Output the predictions for each batch
		forwardPropagations.forEach(fp -> outputBatchPredictions(fp.getOutput(), inceptionV4ClassificationNames));
		
		
	}
	
	private void outputBatchPredictions(NeuronsActivation outputActivation, InceptionV4Labels inceptionV4ClassificationNames){
		Matrix classificationProbablitiesForBatch = outputActivation.getActivations(matrixFactory);
		
		for (int column = 0; column < classificationProbablitiesForBatch.getColumns(); column++) {
			Matrix classificationProbablitiesForExample = classificationProbablitiesForBatch.getColumn(column);
			int predictedIndex = classificationProbablitiesForExample.argmax();
			String predictedClassificationName = inceptionV4ClassificationNames.getLabel(predictedIndex);
			LOGGER.info("Predicted : '" + predictedClassificationName + "' with probability "
					+ classificationProbablitiesForExample.get(predictedIndex));
		}
	}
	
	
}
