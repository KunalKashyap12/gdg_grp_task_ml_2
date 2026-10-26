package com.florascan.backend.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Type-safe configuration properties bound from application.yml (prefix: ml.service).
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "ml.service")
public class MlServiceProperties {

    /**
     * Base URL of the Python FastAPI ML microservice.
     * Default: http://localhost:8000
     */
    private String url = "http://localhost:8000";

    /**
     * Connect and read timeout in milliseconds for the ML service.
     * Default: 10000 ms (10 seconds)
     */
    private int timeoutMs = 10000;

    /**
     * Maximum number of retry attempts if ML service is temporarily unresponsive.
     */
    private int maxRetries = 2;

    /**
     * Mock mode flag to return synthetic predictions when true.
     */
    private boolean mockEnabled = false;
}
