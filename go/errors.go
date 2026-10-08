package von

// VonError is returned for every request failure: transport errors, timeouts,
// encoding errors and non-2xx responses. Inspect it with errors.As:
//
//	var ve *von.VonError
//	if errors.As(err, &ve) && ve.Status == http.StatusUnprocessableEntity { ... }
type VonError struct {
	// Message is the human-readable failure detail. Error returns it verbatim.
	Message string
	// Status is the HTTP status code for server errors, or 0 for transport and
	// timeout failures.
	Status int
	// Details is the raw response body for server errors, when available.
	Details string
}

// Error implements the error interface.
func (e *VonError) Error() string { return e.Message }
