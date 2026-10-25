package com.florascan.backend.controller;

import com.florascan.backend.dto.CropPredictionResponse;
import com.florascan.backend.dto.HealthResponse;
import com.florascan.backend.service.MlServiceClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * Public REST Controller exposing Crop Leaf Disease Diagnostics to Web and Mobile clients.
 */
@RestController
@RequestMapping("/api/crops")
@CrossOrigin(origins = "*", allowedHeaders = "*")
public class CropDiagnosisController {

    private static final Logger log = LoggerFactory.getLogger(CropDiagnosisController.class);

    private final MlServiceClient mlServiceClient;

    public CropDiagnosisController(MlServiceClient mlServiceClient) {
        this.mlServiceClient = mlServiceClient;
    }

    /**
     * Primary Diagnosis Endpoint:
     * Accepts a leaf photo upload, forwards to the Python ML microservice, and returns
     * calibrated botanical species identification and foliar disease diagnosis.
     *
     * @param file     Required leaf image file (.jpg, .jpeg, .png, .webp)
     * @param cropHint Optional user crop confirmation (e.g. "Tomato", "Pepper", "Potato")
     * @param topK     Number of differential candidate diagnoses to return (default: 3)
     * @return CropPredictionResponse
     */
    @PostMapping(value = "/diagnose", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<CropPredictionResponse> diagnoseLeaf(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "crop_hint", required = false) String cropHint,
            @RequestParam(value = "top_k", defaultValue = "3") int topK) {

        String originalFilename = file.getOriginalFilename();
        log.info("Received crop diagnosis request: filename='{}', size={} bytes, cropHint='{}', topK={}",
                originalFilename, file.getSize(), cropHint, topK);

        // Basic client-side validation
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded image cannot be empty (0 bytes).");
        }

        String filename = (originalFilename != null) ? originalFilename.toLowerCase() : "";
        if (!filename.endsWith(".jpg") && !filename.endsWith(".jpeg") && !filename.endsWith(".png") && !filename.endsWith(".webp")) {
            throw new IllegalArgumentException("Unsupported file format. Please upload JPG, PNG, or WEBP.");
        }

        // 1. Call ML microservice
        CropPredictionResponse prediction = mlServiceClient.predictDisease(file, topK, cropHint);

        // 2. (Optional Extension): Save diagnosis to PostgreSQL / MySQL here
        // e.g., diagnosisRepository.save(new CropDiagnosisRecord(...));

        return ResponseEntity.ok(prediction);
    }

    /**
     * Health check verifying end-to-end connectivity from Spring Boot to Python ML engine.
     */
    @GetMapping("/ml-health")
    public ResponseEntity<HealthResponse> checkMlHealth() {
        HealthResponse health = mlServiceClient.checkHealth();
        return ResponseEntity.ok(health);
    }
}
