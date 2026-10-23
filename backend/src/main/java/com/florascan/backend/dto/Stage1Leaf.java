package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Stage 1: Botanical Leaf Species identification metadata and candidate probabilities.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Stage1Leaf {

    @JsonProperty("identified_crop")
    private String identifiedCrop;

    @JsonProperty("confidence")
    private double confidence;

    @JsonProperty("is_conditioned")
    private boolean isConditioned;

    @JsonProperty("candidates")
    private List<CandidateItem> candidates;
}
