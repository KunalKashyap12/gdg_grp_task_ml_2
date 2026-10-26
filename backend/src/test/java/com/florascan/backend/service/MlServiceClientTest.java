package com.florascan.backend.service;

import com.florascan.backend.config.MlServiceProperties;
import com.florascan.backend.dto.CropPredictionResponse;
import com.florascan.backend.dto.PredictionItem;
import com.florascan.backend.dto.TopKItem;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.client.RestTemplate;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MlServiceClientTest {

    private MlServiceClient mlServiceClient;
    private MlServiceProperties properties;

    @BeforeEach
    void setUp() {
        properties = new MlServiceProperties();
        properties.setUrl("http://localhost:8000");
        properties.setMockEnabled(false);
        mlServiceClient = new MlServiceClient(new RestTemplate(), properties);
    }

    @Test
    @DisplayName("Crop Mapping: Map input crop hints to canonical ML names ('corn'->'Maize', 'paddy'->'Rice', 'bell pepper'->'Pepper')")
    void testMapCropHintForMl() {
        assertEquals("Maize", mlServiceClient.mapCropHintForMl("corn"));
        assertEquals("Maize", mlServiceClient.mapCropHintForMl("Corn"));
        assertEquals("Maize", mlServiceClient.mapCropHintForMl("maize"));

        assertEquals("Rice", mlServiceClient.mapCropHintForMl("paddy"));
        assertEquals("Rice", mlServiceClient.mapCropHintForMl("Paddy"));
        assertEquals("Rice", mlServiceClient.mapCropHintForMl("rice"));

        assertEquals("Pepper", mlServiceClient.mapCropHintForMl("bell pepper"));
        assertEquals("Pepper", mlServiceClient.mapCropHintForMl("Bell Pepper"));
        assertEquals("Pepper", mlServiceClient.mapCropHintForMl("pepper"));

        assertEquals("Tomato", mlServiceClient.mapCropHintForMl("tomato"));
        assertNull(mlServiceClient.mapCropHintForMl(null));
        assertNull(mlServiceClient.mapCropHintForMl("   "));
    }

    @Test
    @DisplayName("Crop Mapping: Map ML names back to display names ('Maize'->'Corn', 'Rice'->'Rice (Paddy)', 'Pepper'->'Bell Pepper')")
    void testMapMlCropForDisplay() {
        assertEquals("Corn", mlServiceClient.mapMlCropForDisplay("Maize"));
        assertEquals("Rice (Paddy)", mlServiceClient.mapMlCropForDisplay("Rice"));
        assertEquals("Bell Pepper", mlServiceClient.mapMlCropForDisplay("Pepper"));
        assertEquals("Tomato", mlServiceClient.mapMlCropForDisplay("Tomato"));
        assertNull(mlServiceClient.mapMlCropForDisplay(null));
    }

    @Test
    @DisplayName("Status Handling: Post-processing properly flags and formats all ML statuses")
    void testPostProcessPredictionResponse() {
        // 1. Success
        CropPredictionResponse successResp = CropPredictionResponse.builder()
                .status("success")
                .lowConfidence(false)
                .prediction(PredictionItem.builder().crop("Maize").disease("Common_rust").build())
                .topK(List.of(TopKItem.builder().label("Maize___Common_rust").confidence(0.95).build()))
                .build();
        mlServiceClient.postProcessPredictionResponse(successResp);
        assertEquals("Corn", successResp.getPrediction().getCrop());
        assertFalse(successResp.isLowConfidence());

        // 2. Low confidence
        CropPredictionResponse lowConfResp = CropPredictionResponse.builder()
                .status("low_confidence")
                .threshold(0.74)
                .prediction(PredictionItem.builder().crop("Maize").disease("Common_rust").build())
                .build();
        mlServiceClient.postProcessPredictionResponse(lowConfResp);
        assertTrue(lowConfResp.isLowConfidence());
        assertTrue(lowConfResp.getMessage().contains("AI-assisted indication only"));

        // 3. Not a leaf
        CropPredictionResponse notLeafResp = CropPredictionResponse.builder()
                .status("not_a_leaf")
                .isLeaf(true) // will be set to false
                .build();
        mlServiceClient.postProcessPredictionResponse(notLeafResp);
        assertFalse(notLeafResp.isLeaf());
        assertTrue(notLeafResp.isLowConfidence());
        assertTrue(notLeafResp.getMessage().contains("does not appear to be a botanical plant leaf"));

        // 4. Unsupported crop
        CropPredictionResponse unsuppResp = CropPredictionResponse.builder()
                .status("unsupported_crop")
                .prediction(PredictionItem.builder().crop("Tomato").build())
                .build();
        mlServiceClient.postProcessPredictionResponse(unsuppResp);
        assertTrue(unsuppResp.isLowConfidence());
        assertTrue(unsuppResp.getMessage().contains("Unsupported crop species"));
    }

    @Test
    @DisplayName("Mock Fallback: When mockEnabled=true, returns synthetic prediction without calling network")
    void testMockFallback() {
        properties.setMockEnabled(true);
        MockMultipartFile file = new MockMultipartFile("image", "test.jpg", "image/jpeg", new byte[]{1, 2, 3});
        CropPredictionResponse response = mlServiceClient.predictDisease(file, 5, "corn");

        assertNotNull(response);
        assertEquals("success", response.getStatus());
        assertEquals("Corn", response.getPrediction().getCrop());
        assertEquals("1.0.0-mock", response.getModelVersion());
        assertNotNull(response.getTopK());
        assertEquals(3, response.getTopK().size()); // 3 candidates
    }

    @Test
    @DisplayName("Null or empty file validation")
    void testEmptyFileThrows() {
        MockMultipartFile emptyFile = new MockMultipartFile("image", "test.jpg", "image/jpeg", new byte[0]);
        assertThrows(IllegalArgumentException.class, () -> mlServiceClient.predictDisease(emptyFile, 3, null));
        assertThrows(IllegalArgumentException.class, () -> mlServiceClient.predictDisease(null, 3, null));
    }
}
