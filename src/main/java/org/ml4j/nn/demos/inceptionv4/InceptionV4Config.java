package org.ml4j.nn.demos.inceptionv4;

import java.io.IOException;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasMatrixFactory2;
import org.ml4j.nn.activationfunctions.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.AxonsFactory;
import org.ml4j.nn.axons.AxonsFactoryImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactoryImpl;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilderFactory;
import org.ml4j.nn.components.builders.componentsgraph.DefaultComponents3DGraphBuilderFactory;
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
		return new JBlasMatrixFactory2();
	}

	@Bean
	AxonsFactory axonsFactory() {
		return new AxonsFactoryImpl(matrixFactory());
	}
	
	@Bean
	DirectedAxonsComponentFactory directedAxonsComponentFactory() {
		return new DirectedAxonsComponentFactoryImpl(matrixFactory(), axonsFactory());
	}

	@Bean
	Components3DGraphBuilderFactory components3DGraphBuilderFactory() {
		return new DefaultComponents3DGraphBuilderFactory(directedAxonsComponentFactory());
	}

	@Bean
	DifferentiableActivationFunctionFactory activationFunctionFactory() {
		return new DefaultDifferentiableActivationFunctionFactory();
	}

	@Bean
	SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory() {
		return new DefaultSupervisedFeedForwardNeuralNetworkFactory();
	}
	
	@Bean
	InceptionV4Factory inceptionV4Factory() throws IOException {
		return new DefaultInceptionV4Factory(components3DGraphBuilderFactory(), activationFunctionFactory(),
				supervisedFeedForwardNeuralNetworkFactory(), InceptionV4Demo.class.getClassLoader());
	}
}
