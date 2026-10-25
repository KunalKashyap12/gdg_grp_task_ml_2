================================================================================
FLORASCAN AI — ML SERVICE TO SPRING BOOT BACKEND INTEGRATION GUIDE
================================================================================

Hello! This document contains all the steps, endpoints, and copy-paste Java 
code required to integrate our live Machine Learning model with your Spring 
Boot backend.

You do NOT need to write Python code, install PyTorch, or manage model weights.
Our ML model is already deployed as an ONNX microservice in the cloud.

================================================================================
1. QUICK ACTION CHECKLIST FOR BACKEND DEPLOYMENT (RENDER)
================================================================================

Currently, both services are deployed on Render:
* ML Microservice:  https://gdg-grp-task-ml-2.onrender.com
* Java Backend:     https://crop-backend-r14q.onrender.com

Please complete these 2 quick configurations on your backend Render service:

[ ] ACTION 1: Add Environment Variable in your Render Dashboard
    In your Render dashboard for "crop-backend-r14q", navigate to Environment:
    - Variable Name : ML_SERVICE_URL
    - Value         : https://gdg-grp-task-ml-2.onrender.com

[ ] ACTION 2: Spring Security Configuration
    We tested your live backend and noticed requests currently return:
    {"code":"UNAUTHORIZED","message":"Authentication required."} (HTTP 401).

    - If the leaf diagnosis endpoint is public for guests/frontend:
      Please permit the route in your SecurityConfig.java:
      .requestMatchers("/api/crops/**").permitAll()

    - If diagnosis requires authentication:
      Please share test login credentials or a test JWT token with the team
      so we can test the end-to-end pipeline.


================================================================================
2. LIVE ML SERVICE URLS & DOCUMENTATION
================================================================================

* Base URL:
  https://gdg-grp-task-ml-2.onrender.com

* Interactive Swagger UI (Test in Browser):
  https://gdg-grp-task-ml-2.onrender.com/docs

* Alternative ReDoc Documentation:
  https://gdg-grp-task-ml-2.onrender.com/redoc

* Service Health Check Endpoint:
  GET https://gdg-grp-task-ml-2.onrender.com/api/v1/health

* Prediction / Diagnosis Endpoint:
  POST https://gdg-grp-task-ml-2.onrender.com/api/v1/predict

* Supported Crops & Diseases Registry:
  GET https://gdg-grp-task-ml-2.onrender.com/api/v1/classes


================================================================================
3. ML PREDICTION API SPECIFICATION (POST /api/v1/predict)
================================================================================

When a user in your frontend or mobile app uploads a leaf photo, your backend 
receives the image and forwards it to our ML API.

* HTTP Method: POST
* URL: https://gdg-grp-task-ml-2.onrender.com/api/v1/predict
* Content-Type: multipart/form-data

Request Form Fields:
1. "image" (File, REQUIRED)
   - The binary image file of the crop leaf photo.
   - Supported formats: .jpg, .jpeg, .png, .webp
   - Max file size: 5 MB
   - (Note: field name "file" is also accepted as fallback)

2. "crop" (String, OPTIONAL)
   - User-selected or known plant species hint (e.g. "Tomato", "Potato", "Apple").
   - If omitted or empty, the model auto-identifies the plant species.
   - Supported crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, 
                      Peach, Pepper, Potato, Raspberry, Soybean, Squash, 
                      Strawberry, Tomato.


================================================================================
4. EXPECTED JSON RESPONSE (200 OK)
================================================================================

{
  "status": "success",
  "is_leaf": true,
  "prediction": {
    "class_id": 13,
    "crop": "Pepper",
    "disease": "Bacterial_spot",
    "label": "Pepper___Bacterial_spot",
    "confidence": 0.9674
  },
  "top_k": [
    { "label": "Pepper___Bacterial_spot", "confidence": 0.9674 },
    { "label": "Pepper___Healthy", "confidence": 0.0215 },
    { "label": "Tomato___Bacterial_spot", "confidence": 0.0081 }
  ],
  "low_confidence": false,
  "threshold": 0.74,
  "message": "Stage 1: Pepper leaf identified (97.6%). Stage 2: Bacterial spot diagnosed (96.7% confidence).",
  "model_version": "1.0.0",
  "inference_time_ms": 18.2,
  "crop_conditioned": null,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}

KEY RESPONSE FIELDS:
- "status": "success" (confident), "low_confidence" (blurry/uncertain), or "not_a_leaf" (non-leaf object uploaded).
- "is_leaf": Boolean indicating if the uploaded image is genuinely a plant leaf.
- "prediction": Top diagnosis containing crop name, disease name, and calibrated confidence (0.0 to 1.0).
- "top_k": Top 3 candidate diagnoses with individual confidence scores.
- "low_confidence": Boolean flag. If true, advise the user to retake a clearer, well-lit photo.
- "message": Human-readable diagnostic assessment summary.
- "disclaimer": Mandatory agricultural advisory statement to display in your UI.


================================================================================
5. ERROR RESPONSES & STATUS CODES
================================================================================

All error responses return a standardized JSON structure:
{"error": {"code": "ERROR_CODE", "message": "Detailed explanation"}}

Status Codes:
* 400 Bad Request:
  File is corrupted, empty, or unreadable as an image.
* 413 Payload Too Large:
  Image exceeds the 5 MB limit.
* 415 Unsupported Media Type:
  Uploaded file is not JPEG, PNG, or WebP.
* 422 Unprocessable Entity:
  Missing required form field "image".
* 500 Internal Server Error:
  Inference execution error.


================================================================================
6. COMPLETE JAVA + SPRING BOOT IMPLEMENTATION (COPY-PASTE READY)
================================================================================

You can copy the code below directly into your Spring Boot project. 
(All these exact Java files are also available in the "backend/" folder 
of our ML repository).

--------------------------------------------------------------------------------
STEP A: Maven Dependencies (pom.xml)
--------------------------------------------------------------------------------
Ensure your pom.xml contains:

<dependencies>
    <!-- Core Spring Boot Web Starter (provides RestTemplate & Spring MVC) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>

    <!-- Jackson JSON Databind (typically included by default) -->
    <dependency>
        <groupId>com.fasterxml.jackson.core</groupId>
        <artifactId>jackson-databind</artifactId>
    </dependency>

    <!-- Lombok (optional, for @Data, @Builder) -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
</dependencies>


--------------------------------------------------------------------------------
STEP B: Configuration (application.yml or application.properties)
--------------------------------------------------------------------------------
Add this to src/main/resources/application.yml:

ml:
  service:
    url: ${ML_SERVICE_URL:https://gdg-grp-task-ml-2.onrender.com}
    timeout-ms: 30000

(Note: Set timeout to 30000ms [30s] to accommodate cloud instance cold starts).


--------------------------------------------------------------------------------
STEP C: Java DTO Classes
--------------------------------------------------------------------------------

// File: PredictionItem.java
package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class PredictionItem {
    @JsonProperty("class_id")
    private int classId;
    private String crop;
    private String disease;
    private String label;
    private double confidence;
}


// File: TopKItem.java
package com.florascan.backend.dto;

import lombok.Data;

@Data
public class TopKItem {
    private String label;
    private double confidence;
}


// File: CropPredictionResponse.java
package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class CropPredictionResponse {
    private String status;
    
    @JsonProperty("is_leaf")
    private boolean isLeaf;
    
    private PredictionItem prediction;
    
    @JsonProperty("top_k")
    private List<TopKItem> topK;
    
    @JsonProperty("low_confidence")
    private boolean lowConfidence;
    
    private double threshold;
    private String message;
    
    @JsonProperty("model_version")
    private String modelVersion;
    
    @JsonProperty("inference_time_ms")
    private double inferenceTimeMs;
    
    private String disclaimer;
}


--------------------------------------------------------------------------------
STEP D: ML Service Client (MlServiceClient.java)
--------------------------------------------------------------------------------
package com.florascan.backend.service;

import com.florascan.backend.dto.CropPredictionResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.Duration;

@Service
public class MlServiceClient {

    private final RestTemplate restTemplate;
    private final String mlServiceUrl;

    public MlServiceClient(
            RestTemplateBuilder builder,
            @Value("${ml.service.url:https://gdg-grp-task-ml-2.onrender.com}") String mlServiceUrl,
            @Value("${ml.service.timeout-ms:30000}") long timeoutMs) {
        this.mlServiceUrl = mlServiceUrl;
        this.restTemplate = builder
                .setConnectTimeout(Duration.ofMillis(timeoutMs))
                .setReadTimeout(Duration.ofMillis(timeoutMs))
                .build();
    }

    /**
     * Forwards uploaded leaf image to the Python FastAPI ML microservice.
     *
     * @param imageFile The uploaded leaf photo file from user
     * @param cropHint  Optional user-selected crop hint (e.g. "Tomato", "Potato")
     * @return CropPredictionResponse containing top diagnosis, confidence & top-k
     */
    public CropPredictionResponse predictDisease(MultipartFile imageFile, String cropHint) throws IOException {
        String targetUrl = mlServiceUrl + "/api/v1/predict";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        // Wrap file bytes with original filename preserved
        ByteArrayResource fileResource = new ByteArrayResource(imageFile.getBytes()) {
            @Override
            public String getFilename() {
                String original = imageFile.getOriginalFilename();
                return (original != null && !original.isBlank()) ? original : "leaf.jpg";
            }
        };

        body.add("image", fileResource);

        if (cropHint != null && !cropHint.isBlank()) {
            body.add("crop", cropHint.trim());
        }

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<CropPredictionResponse> response = restTemplate.postForEntity(
                targetUrl,
                requestEntity,
                CropPredictionResponse.class
        );

        return response.getBody();
    }
}


--------------------------------------------------------------------------------
STEP E: REST Controller (CropDiagnosisController.java)
--------------------------------------------------------------------------------
package com.florascan.backend.controller;

import com.florascan.backend.dto.CropPredictionResponse;
import com.florascan.backend.service.MlServiceClient;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/crops")
@CrossOrigin(origins = "*")
public class CropDiagnosisController {

    private final MlServiceClient mlServiceClient;

    public CropDiagnosisController(MlServiceClient mlServiceClient) {
        this.mlServiceClient = mlServiceClient;
    }

    /**
     * Public endpoint for frontend / mobile app to upload leaf photo.
     * POST /api/crops/diagnose
     */
    @PostMapping(value = "/diagnose", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<CropPredictionResponse> diagnoseLeaf(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "crop", required = false) String crop) throws IOException {

        if (image == null || image.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        CropPredictionResponse result = mlServiceClient.predictDisease(image, crop);

        // OPTIONAL: Save diagnosis record in your database here (PostgreSQL / MySQL):
        // diagnosisRepository.save(new DiagnosisRecord(userId, result.getPrediction().getDisease(), ...));

        return ResponseEntity.ok(result);
    }
}


================================================================================
7. END-TO-END TESTING WITH CURL
================================================================================

A. Test ML Service Directly:
curl -X POST "https://gdg-grp-task-ml-2.onrender.com/api/v1/predict" \
     -F "image=@sample_leaf.jpg" \
     -F "crop=Tomato"

B. Test Your Backend (once deployed):
curl -X POST "https://crop-backend-r14q.onrender.com/api/crops/diagnose" \
     -F "image=@sample_leaf.jpg" \
     -F "crop=Tomato"


================================================================================
8. PRODUCTION TIP: RENDER COLD STARTS & KEEP-ALIVE
================================================================================

Because free Render instances spin down after 15 minutes of inactivity:
1. The first request after sleeping may take ~30-45 seconds to spin up.
2. We recommend setting up a free cron job or uptime pinger (e.g., cron-job.org 
   or uptimerobot.com) to ping:
   GET https://gdg-grp-task-ml-2.onrender.com/api/v1/health
   and
   GET https://crop-backend-r14q.onrender.com/actuator/health
   every 10 minutes. This keeps both services warm and responsive 24/7!

================================================================================
If you have any questions or need extra response fields, let us know!
================================================================================
