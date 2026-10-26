package com.florascan.backend.controller;

import com.florascan.backend.dto.CropPredictionResponse;
import com.florascan.backend.dto.HealthResponse;
import com.florascan.backend.dto.PredictionItem;
import com.florascan.backend.exception.GlobalExceptionHandler;
import com.florascan.backend.service.MlServiceClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(MockitoExtension.class)
class CropDiagnosisControllerTest {

    private MockMvc mockMvc;

    @Mock
    private MlServiceClient mlServiceClient;

    @InjectMocks
    private CropDiagnosisController controller;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    @DisplayName("Diagnose: Valid image upload under 5 MB succeeds")
    void testValidImageUpload() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file", "leaf.jpg", MediaType.IMAGE_JPEG_VALUE, new byte[1024]);

        CropPredictionResponse mockResponse = CropPredictionResponse.builder()
                .status("success")
                .isLeaf(true)
                .prediction(PredictionItem.builder().crop("Corn").disease("Common_rust").confidence(0.95).build())
                .build();

        when(mlServiceClient.predictDisease(any(), any(), any())).thenReturn(mockResponse);

        mockMvc.perform(multipart("/api/crops/diagnose")
                        .file(file)
                        .param("crop_hint", "corn")
                        .param("top_k", "3"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("success"))
                .andExpect(jsonPath("$.prediction.crop").value("Corn"));
    }

    @Test
    @DisplayName("Upload Size: Files exceeding 5 MB return HTTP 413 Payload Too Large with clear message")
    void testOversizedUploadReturns413() throws Exception {
        // 6 MB file
        byte[] oversizedBytes = new byte[6 * 1024 * 1024];
        MockMultipartFile file = new MockMultipartFile(
                "file", "large_leaf.jpg", MediaType.IMAGE_JPEG_VALUE, oversizedBytes);

        mockMvc.perform(multipart("/api/crops/diagnose").file(file))
                .andExpect(status().isPayloadTooLarge())
                .andExpect(jsonPath("$.error.code").value("FILE_TOO_LARGE"))
                .andExpect(jsonPath("$.error.message").value("Uploaded image exceeds the maximum permitted size of 5 MB."));
    }

    @Test
    @DisplayName("Format Validation: Unsupported formats return HTTP 400 Bad Request")
    void testUnsupportedFormatReturns400() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file", "document.pdf", MediaType.APPLICATION_PDF_VALUE, new byte[100]);

        mockMvc.perform(multipart("/api/crops/diagnose").file(file))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error.code").value("INVALID_INPUT"));
    }

    @Test
    @DisplayName("Health Check: Delegates to ML Service checkHealth and returns metadata")
    void testHealthCheck() throws Exception {
        HealthResponse health = HealthResponse.builder()
                .status("healthy")
                .modelLoaded(true)
                .numClasses(37)
                .supportedCropsCount(10)
                .calibrationActive(true)
                .temperature(0.4792)
                .disclaimer("AI-assisted indication only")
                .build();

        when(mlServiceClient.checkHealth()).thenReturn(health);

        mockMvc.perform(get("/api/crops/ml-health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("healthy"))
                .andExpect(jsonPath("$.model_loaded").value(true))
                .andExpect(jsonPath("$.num_classes").value(37));
    }
}
