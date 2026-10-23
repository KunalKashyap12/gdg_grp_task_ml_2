package com.florascan.backend.config;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

/**
 * Spring configuration providing pre-configured RestTemplate with timeouts.
 */
@Configuration
public class MlServiceConfig {

    private final MlServiceProperties properties;

    public MlServiceConfig(MlServiceProperties properties) {
        this.properties = properties;
    }

    @Bean
    public RestTemplate mlRestTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(Duration.ofMillis(properties.getTimeoutMs()))
                .setReadTimeout(Duration.ofMillis(properties.getTimeoutMs()))
                .build();
    }
}
