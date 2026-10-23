package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Candidate prediction item for Top-K differential diagnoses.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TopKItem {

    @JsonProperty("label")
    private String label;

    @JsonProperty("confidence")
    private double confidence;
}
