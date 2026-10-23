package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Ranked candidate for botanical species or specific disease diagnoses.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class CandidateItem {

    @JsonProperty("crop")
    private String crop;

    @JsonProperty("disease")
    private String disease;

    @JsonProperty("label")
    private String label;

    @JsonProperty("confidence")
    private double confidence;
}
