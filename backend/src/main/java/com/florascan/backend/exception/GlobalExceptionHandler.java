package com.florascan.backend.exception;

import com.florascan.backend.dto.ErrorResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(MlServiceException.class)
    public ResponseEntity<ErrorResponse> handleMlServiceException(MlServiceException ex) {
        ErrorResponse errorResponse = ErrorResponse.builder()
                .error(ErrorResponse.ErrorDetail.builder()
                        .code(ex.getErrorCode())
                        .message(ex.getMessage())
                        .build())
                .build();
        HttpStatus status = ex.getHttpStatus() != null ? ex.getHttpStatus() : HttpStatus.INTERNAL_SERVER_ERROR;
        return ResponseEntity.status(status.value()).body(errorResponse);
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ErrorResponse> handleMaxUploadSizeExceeded(MaxUploadSizeExceededException ex) {
        ErrorResponse errorResponse = ErrorResponse.builder()
                .error(ErrorResponse.ErrorDetail.builder()
                        .code("FILE_TOO_LARGE")
                        .message("Uploaded image exceeds the maximum permitted size of 5 MB.")
                        .build())
                .build();
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE).body(errorResponse);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
        ErrorResponse errorResponse = ErrorResponse.builder()
                .error(ErrorResponse.ErrorDetail.builder()
                        .code("INVALID_INPUT")
                        .message(ex.getMessage())
                        .build())
                .build();
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneralException(Exception ex) {
        ErrorResponse errorResponse = ErrorResponse.builder()
                .error(ErrorResponse.ErrorDetail.builder()
                        .code("INTERNAL_SERVER_ERROR")
                        .message("An unexpected server error occurred: " + ex.getMessage())
                        .build())
                .build();
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
    }
}
