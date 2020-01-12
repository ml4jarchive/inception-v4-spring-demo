package org.ml4j.nn.demos.inceptionv4;

import java.io.IOException;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilderFactory;
import org.ml4j.nn.components.builders.componentsgraph.DefaultComponents3DGraphBuilderFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.impl.DefaultInceptionV4Factory;
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
	DirectedComponentFactory directedAxonsComponentFactory() {
		return new DefaultDirectedComponentFactoryImpl(matrixFactory(), axonsFactory(), activationFunctionFactory());
	}

	@Bean
	Components3DGraphBuilderFactory<DefaultChainableDirectedComponent<?, ?>> components3DGraphBuilderFactory() {
		return new DefaultComponents3DGraphBuilderFactory<>(directedAxonsComponentFactory());
	}

	@Bean
	DifferentiableActivationFunctionFactory activationFunctionFactory() {
		return new DefaultDifferentiableActivationFunctionFactory();
	}

	@Bean
	SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory() {
		return new DefaultSupervisedFeedForwardNeuralNetworkFactory(directedAxonsComponentFactory());
	}
	
	@Bean
	InceptionV4Factory inceptionV4Factory() throws IOException {
		return new DefaultInceptionV4Factory(components3DGraphBuilderFactory(), matrixFactory(), 
				supervisedFeedForwardNeuralNetworkFactory(), InceptionV4Demo.class.getClassLoader());
	}
}
