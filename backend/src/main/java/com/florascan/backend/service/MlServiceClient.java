package com.florascan.backend.service;

import com.florascan.backend.config.MlServiceProperties;
import com.florascan.backend.dto.*;
import com.florascan.backend.exception.MlServiceException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
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
import java.util.Collections;
import java.util.List;

/**
 * Robust Client Service responsible for HTTP communication with the Python FastAPI ML microservice.
 */
@Service
public class MlServiceClient {

    private static final Logger log = LoggerFactory.getLogger(MlServiceClient.class);

    private final RestTemplate restTemplate;
    private final MlServiceProperties properties;

    @Value("${ml.mock.enabled:false}")
    private boolean mockEnabled;

    public MlServiceClient(RestTemplate mlRestTemplate, MlServiceProperties properties) {
        this.restTemplate = mlRestTemplate;
        this.properties = properties;
    }

    /**
     * Map user/client crop names to canonical ML botanical labels before invoking the ML microservice.
     * "corn" -> "Maize", "paddy" -> "Rice", "bell pepper" -> "Pepper"
     */
    public String mapCropHintForMl(String cropHint) {
        if (cropHint == null || cropHint.isBlank()) {
            return null;
        }
        String clean = cropHint.trim().toLowerCase();
        return switch (clean) {
            case "corn", "maize" -> "Maize";
            case "paddy", "rice" -> "Rice";
            case "bell pepper", "bell_pepper", "bellpepper", "pepper" -> "Pepper";
            case "apple" -> "Apple";
            case "cherry" -> "Cherry";
            case "grape" -> "Grape";
            case "peach" -> "Peach";
            case "potato" -> "Potato";
            case "strawberry" -> "Strawberry";
            case "tomato" -> "Tomato";
            default -> {
                if (clean.length() > 1) {
                    yield Character.toUpperCase(clean.charAt(0)) + clean.substring(1);
                }
                yield cropHint.trim();
            }
        };
    }

    /**
     * Map internal ML botanical labels back to user-friendly crop display names.
     * "Maize" -> "Corn", "Rice" -> "Rice (Paddy)", "Pepper" -> "Bell Pepper"
     */
    public String mapMlCropForDisplay(String mlCrop) {
        if (mlCrop == null) {
            return null;
        }
        return switch (mlCrop) {
            case "Maize" -> "Corn";
            case "Rice" -> "Rice (Paddy)";
            case "Pepper" -> "Bell Pepper";
            default -> mlCrop;
        };
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

        boolean isMock = this.mockEnabled || properties.isMockEnabled();
        if (isMock) {
            log.info("Mock mode active (ml.mock.enabled=true). Returning synthetic prediction without network call.");
            return buildMockPrediction(cropHint);
        }

        String targetUrl = getBaseUrl() + "/api/v1/predict";
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

            // ML microservice hardcodes 3 candidates and ignores top_k; do not assume a larger number
            int requestedTopK = (topK != null && topK > 0) ? topK : 3;
            body.add("top_k", requestedTopK);

            // Map crop names before calling ML service ("corn" -> "Maize", "paddy" -> "Rice", etc.)
            String mappedCropHint = mapCropHintForMl(cropHint);
            if (mappedCropHint != null && !mappedCropHint.isBlank()) {
                body.add("crop", mappedCropHint);
            }

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            long startTime = System.currentTimeMillis();
            ResponseEntity<CropPredictionResponse> responseEntity = restTemplate.postForEntity(
                    targetUrl,
                    requestEntity,
                    CropPredictionResponse.class
            );
            long duration = System.currentTimeMillis() - startTime;

            log.info("Received ML response in {} ms with status code: {}", duration, responseEntity.getStatusCode());
            CropPredictionResponse response = responseEntity.getBody();

            // Post-process response: normalize messages, status semantics, and display crop names
            postProcessPredictionResponse(response);

            return response;

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
     * Post-processes prediction response:
     * - Maps ML statuses (success, low_confidence, not_a_leaf, unsupported_crop) to clear user messages
     * - Never presents low_confidence or unsupported_crop as confirmed diagnoses
     * - Maps ML botanical crop names back to display names ("Maize" -> "Corn", etc.)
     * - Reads candidates returned without assuming a larger count than ML actually provides
     */
    public void postProcessPredictionResponse(CropPredictionResponse response) {
        if (response == null) {
            return;
        }

        String rawStatus = response.getStatus() != null ? response.getStatus().toLowerCase() : "unknown";

        switch (rawStatus) {
            case "success" -> {
                if (response.getMessage() == null || response.getMessage().isBlank()) {
                    response.setMessage("Disease identified successfully with high calibrated confidence.");
                }
            }
            case "low_confidence" -> {
                response.setLowConfidence(true);
                response.setMessage("AI-assisted indication only: Diagnostic confidence is low. Please retake a clear, well-lit close-up photo of a single leaf against a neutral background. Do not treat this as a confirmed agricultural diagnosis.");
            }
            case "not_a_leaf" -> {
                response.setLeaf(false);
                response.setLowConfidence(true);
                response.setMessage("The uploaded image was rejected because it does not appear to be a botanical plant leaf. Please upload a clear photo of a crop leaf.");
            }
            case "unsupported_crop" -> {
                response.setLowConfidence(true);
                response.setMessage("Unsupported crop species detected. FloraScan currently supports: Apple, Cherry, Grape, Maize (Corn), Peach, Pepper, Potato, Rice, Strawberry, and Tomato. Do not use this indication as an agricultural diagnosis.");
            }
            default -> {
                if (response.isLowConfidence()) {
                    response.setMessage("AI-assisted indication only: Diagnostic confidence is low. Not a confirmed agricultural diagnosis.");
                }
            }
        }

        // Map ML crop names back for display
        if (response.getPrediction() != null && response.getPrediction().getCrop() != null) {
            response.getPrediction().setCrop(mapMlCropForDisplay(response.getPrediction().getCrop()));
        }

        if (response.getStage1Leaf() != null) {
            if (response.getStage1Leaf().getIdentifiedCrop() != null) {
                response.getStage1Leaf().setIdentifiedCrop(mapMlCropForDisplay(response.getStage1Leaf().getIdentifiedCrop()));
            }
            if (response.getStage1Leaf().getCandidates() != null) {
                for (CandidateItem cand : response.getStage1Leaf().getCandidates()) {
                    if (cand.getCrop() != null) {
                        cand.setCrop(mapMlCropForDisplay(cand.getCrop()));
                    }
                }
            }
        }

        // Safe reading of top_k candidates: log or retain actual candidates returned without assuming count > actual
        List<TopKItem> topKItems = response.getTopK();
        if (topKItems == null) {
            response.setTopK(Collections.emptyList());
        }
    }

    /**
     * Builds a realistic mock CropPredictionResponse when mock mode is enabled.
     */
    public CropPredictionResponse buildMockPrediction(String cropHint) {
        String canonicalCrop = (cropHint != null && !cropHint.isBlank()) ? mapCropHintForMl(cropHint) : "Tomato";
        String displayCrop = mapMlCropForDisplay(canonicalCrop);
        String disease = "Early_blight";
        String label = canonicalCrop + "___" + disease;

        return CropPredictionResponse.builder()
                .status("success")
                .isLeaf(true)
                .prediction(PredictionItem.builder()
                        .classId(1)
                        .crop(displayCrop)
                        .disease(disease)
                        .label(label)
                        .confidence(0.924)
                        .build())
                .topK(List.of(
                        TopKItem.builder().label(label).confidence(0.924).build(),
                        TopKItem.builder().label(canonicalCrop + "___Healthy").confidence(0.052).build(),
                        TopKItem.builder().label(canonicalCrop + "___Septoria_leaf_spot").confidence(0.018).build()
                ))
                .lowConfidence(false)
                .threshold(0.74)
                .message("Mock diagnosis generated for testing.")
                .modelVersion("1.0.0-mock")
                .inferenceTimeMs(15.0)
                .disclaimer("AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.")
                .build();
    }

    /**
     * Checks liveness and health status of the ML service.
     * Keep calling GET /health (returns metadata HealthResponse.java expects).
     *
     * @return HealthResponse with model status and calibration metadata
     */
    public HealthResponse checkHealth() {
        String healthUrl = getBaseUrl() + "/health";
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

    private String getBaseUrl() {
        String url = properties.getUrl();
        if (url == null || url.isBlank()) {
            return "http://localhost:8000";
        }
        return url.endsWith("/") ? url.substring(0, url.length() - 1) : url;
    }
}
