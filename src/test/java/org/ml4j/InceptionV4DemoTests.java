package org.ml4j;

import org.junit.jupiter.api.Test;
import org.ml4j.nn.demos.inceptionv4.InceptionV4Config;
import org.ml4j.nn.demos.inceptionv4.InceptionV4Demo;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(classes= { InceptionV4Config.class, InceptionV4Demo.class })
class InceptionV4DemoTests {

	@Test
	void contextLoads() {
	}

}
