package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Stage 2: Foliar Pathology Diagnosis conditional on identified or confirmed leaf species.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Stage2Disease {

    @JsonProperty("disease")
    private String disease;

    @JsonProperty("confidence")
    private double confidence;

    @JsonProperty("is_healthy")
    private boolean isHealthy;

    @JsonProperty("pathology_mass")
    private double pathologyMass;

    @JsonProperty("candidates")
    private List<CandidateItem> candidates;
}
