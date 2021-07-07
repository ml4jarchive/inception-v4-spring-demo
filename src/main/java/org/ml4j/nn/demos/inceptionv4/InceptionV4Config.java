package org.ml4j.nn.demos.inceptionv4;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.datasets.images.DirectoryImagesWithPathsDataSet;
import org.ml4j.nn.datasets.images.ImagesDataSet;
import org.ml4j.nn.datasets.images.LabeledImagesDataSet;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.impl.DefaultInceptionV4Factory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactoryImpl;
import org.ml4j.nn.supervised.DefaultSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class InceptionV4Config {

	@Bean
	MatrixFactory matrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Bean
	AxonsFactory axonsFactory() {
		return new DefaultAxonsFactoryImpl(matrixFactory());
	}
	
	@Bean
	DirectedComponentFactory directedComponentFactory() {
		return new DefaultDirectedComponentFactoryImpl(matrixFactory(), axonsFactory(), activationFunctionFactory(), 
				directedComponentsContext());
	}
	
	@Bean
	DirectedComponentsContext directedComponentsContext() {
		return new DirectedComponentsContextImpl(matrixFactory(), false);
	}

	@Bean
	DefaultSessionFactory sessionFactory() {
		return new DefaultSessionFactoryImpl(matrixFactory(), 
				directedComponentFactory(), null,  // No DirectedLayerFactory needed for this demo.
				supervisedFeedForwardNeuralNetworkFactory(), directedComponentsContext());
	}

	@Bean
	DifferentiableActivationFunctionFactory activationFunctionFactory() {
		return new DefaultDifferentiableActivationFunctionFactory();
	}

	@Bean
	SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory() {
		return new DefaultSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory());
	}
	
	@Bean
	InceptionV4Factory inceptionV4Factory() throws IOException {
		return new DefaultInceptionV4Factory(sessionFactory(), matrixFactory(), 
				InceptionV4Demo.class.getClassLoader());
	}
	
	@Bean("filenameLabeledImagesDataSet")
	LabeledImagesDataSet<String> filenameLabeledImagesDataSet() {

		// Create the data set
		// Define images Directory
		Path imagesDirectory = new File(InceptionV4Demo.class.getClassLoader().getResource("test_images").getFile())
				.toPath();

		return new DirectoryImagesWithPathsDataSet(imagesDirectory, path -> true, 299, 299)
				.getLabeledImagesDataSet((index, pathName) -> pathName.getParent().getFileName().toString());
	}
	
	@Bean("imagesDataSet")
	ImagesDataSet imagesDataSet() {
		return filenameLabeledImagesDataSet().getDataSet();
	}
}
