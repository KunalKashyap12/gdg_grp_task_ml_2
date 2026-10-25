package com.florascan.backend.service;

import com.florascan.backend.config.MlServiceProperties;
import com.florascan.backend.dto.CropPredictionResponse;
import com.florascan.backend.dto.HealthResponse;
import com.florascan.backend.exception.MlServiceException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * Robust Client Service responsible for HTTP communication with the Python FastAPI ML microservice.
 */
@Service
public class MlServiceClient {

    private static final Logger log = LoggerFactory.getLogger(MlServiceClient.class);

    private final RestTemplate restTemplate;
    private final MlServiceProperties properties;

    public MlServiceClient(RestTemplate mlRestTemplate, MlServiceProperties properties) {
        this.restTemplate = mlRestTemplate;
        this.properties = properties;
    }

    /**
     * Sends an uploaded leaf photograph to the ML service for botanical and disease diagnosis.
     *
     * @param imageFile Uploaded MultipartFile image
     * @param topK      Optional number of top candidate diagnoses (default: 3)
     * @param cropHint  Optional user-confirmed crop (e.g. "Tomato", "Pepper", "Potato")
     * @return CropPredictionResponse containing calibrated diagnosis
     */
    public CropPredictionResponse predictDisease(MultipartFile imageFile, Integer topK, String cropHint) {
        if (imageFile == null || imageFile.isEmpty()) {
            throw new IllegalArgumentException("Uploaded image file cannot be null or empty.");
        }

        String targetUrl = properties.getUrl() + "/api/v1/predict";
        log.info("Dispatching leaf inference request to ML Service: {}", targetUrl);

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            // Wrap file bytes in ByteArrayResource with filename preservation
            ByteArrayResource fileResource = new ByteArrayResource(imageFile.getBytes()) {
                @Override
                public String getFilename() {
                    String original = imageFile.getOriginalFilename();
                    return (original != null && !original.isBlank()) ? original : "leaf.jpg";
                }
            };

            body.add("image", fileResource);
            body.add("top_k", topK != null ? topK : 3);

            if (cropHint != null && !cropHint.isBlank()) {
                body.add("crop", cropHint.trim());
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            long startTime = System.currentTimeMillis();
            ResponseEntity<CropPredictionResponse> response = restTemplate.postForEntity(
                    targetUrl,
                    requestEntity,
                    CropPredictionResponse.class
            );
            long duration = System.currentTimeMillis() - startTime;

            log.info("Received ML response in {} ms with status code: {}", duration, response.getStatusCode());
            return response.getBody();

        } catch (HttpClientErrorException ex) {
            log.error("ML service rejected request (4xx): {} - {}", ex.getStatusCode(), ex.getResponseBodyAsString());
            HttpStatus status = HttpStatus.resolve(ex.getStatusCode().value());
            throw new MlServiceException(
                    "ML validation error: " + ex.getResponseBodyAsString(),
                    status != null ? status : HttpStatus.BAD_REQUEST,
                    "ML_CLIENT_ERROR"
            );
        } catch (HttpServerErrorException ex) {
            log.error("ML service internal error (5xx): {} - {}", ex.getStatusCode(), ex.getResponseBodyAsString());
            throw new MlServiceException(
                    "ML inference failure: " + ex.getResponseBodyAsString(),
                    HttpStatus.BAD_GATEWAY,
                    "ML_SERVER_ERROR"
            );
        } catch (ResourceAccessException ex) {
            log.error("Failed to connect to ML service at {}: {}", targetUrl, ex.getMessage());
            throw new MlServiceException(
                    "ML inference service is currently unavailable or unreachable at " + properties.getUrl(),
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "ML_SERVICE_UNAVAILABLE"
            );
        } catch (IOException ex) {
            log.error("Failed to read bytes from uploaded MultipartFile: {}", ex.getMessage());
            throw new MlServiceException(
                    "Unable to read image bytes: " + ex.getMessage(),
                    HttpStatus.BAD_REQUEST,
                    "IMAGE_READ_ERROR"
            );
        }
    }

    /**
     * Checks liveness and health status of the ML service.
     *
     * @return HealthResponse with model status and calibration metadata
     */
    public HealthResponse checkHealth() {
        String healthUrl = properties.getUrl() + "/health";
        try {
            return restTemplate.getForObject(healthUrl, HealthResponse.class);
        } catch (Exception ex) {
            log.warn("ML health check failed at {}: {}", healthUrl, ex.getMessage());
            return HealthResponse.builder()
                    .status("unreachable")
                    .modelLoaded(false)
                    .build();
        }
    }
}
