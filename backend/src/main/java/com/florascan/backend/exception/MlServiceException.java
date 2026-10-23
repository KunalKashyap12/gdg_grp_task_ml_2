package com.florascan.backend.exception;

import org.springframework.http.HttpStatus;

/**
 * Custom runtime exception thrown when communicating with the ML microservice.
 */
public class MlServiceException extends RuntimeException {

    private final HttpStatus httpStatus;
    private final String errorCode;

    public MlServiceException(String message, HttpStatus httpStatus, String errorCode) {
        super(message);
        this.httpStatus = httpStatus;
        this.errorCode = errorCode;
    }

    public MlServiceException(String message, Throwable cause, HttpStatus httpStatus, String errorCode) {
        super(message, cause);
        this.httpStatus = httpStatus;
        this.errorCode = errorCode;
    }

    public HttpStatus getHttpStatus() {
        return httpStatus;
    }

    public String getErrorCode() {
        return errorCode;
    }
}
