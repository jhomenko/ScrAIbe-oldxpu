            except RuntimeError as e:
                error_msg = str(e)
                # Simplified XPU error handling: retry on CPU float32 for type mismatches
                if "Input type" in error_msg and "bias type" in error_msg:
                    if self.verbose:
                        print("Type mismatch on XPU, retrying on CPU float32")
                    self.model = self.model.to("cpu")
                    input_features = input_features.to("cpu", dtype=torch.float32)
                    if "decoder_input_ids" in generate_kwargs:
                        generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to("cpu")
                    generated_ids = self.model.generate(input_features, **generate_kwargs)
                else:
                    # Re-raise other runtime errors
                    raise
                
                # Handle type mismatch errors
                if "Input type" in error_msg and "bias type" in error_msg:
                    if "BFloat16" in error_msg:
                        if self.verbose:
                            print("Type mismatch detected. Converting inputs to BFloat16.")
                        # Hybrid approach: Keep on CPU for conversion, then move to target device
                        input_features = input_features.to("cpu", dtype=torch.float32)
                        input_features = input_features.to(model_device, dtype=torch.bfloat16)
                        
                        if "decoder_input_ids" in generate_kwargs:
                            generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to(model_device)
                        
                        # Try generation with explicit BFloat16
                        generated_ids = self.model.generate(
                            input_features,
                            **generate_kwargs
                        )
                    else:
                        # For other type mismatches, try CPU fallback
                        if self.verbose:
                            print(f"Type mismatch: {error_msg}. Falling back to CPU.")
                        self.model = self.model.to("cpu")
                        input_features = input_features.to("cpu", dtype=torch.float32)
                        
                        if "decoder_input_ids" in generate_kwargs:
                            generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to("cpu")
                        
                        # Generate on CPU
                        generated_ids = self.model.generate(
                            input_features,
                            **generate_kwargs
                        )
                
                # Handle engine creation errors
                elif "could not create an engine" in error_msg and model_device.type == "xpu":
                    if self.verbose:
                        print("XPU fallback: could not create engine, falling back to CPU float32.")
                    # CPU float32 fallback
                    self.model = self.model.to("cpu", dtype=torch.float32)
                    input_features = input_features.to("cpu", dtype=torch.float32)
                    if "decoder_input_ids" in generate_kwargs:
                        generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to("cpu")
                    generated_ids = self.model.generate(input_features, **generate_kwargs)
                    self.model = self.model.to("cpu", dtype=torch.float32)
                    input_features = input_features.to("cpu", dtype=torch.float32)
                    if "decoder_input_ids" in generate_kwargs:
                        generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to("cpu")
                    generated_ids = self.model.generate(
                        input_features,
                        **generate_kwargs
                    )
                    
                    # Try with explicit BFloat16 if model is using it
                    if model_dtype == torch.bfloat16:
                        try:
                            # Hybrid approach: Keep on CPU for conversion, then move to target device
                            input_features = input_features.to("cpu", dtype=torch.float32)
                            input_features = input_features.to(model_device, dtype=torch.bfloat16)
                            
                            if "decoder_input_ids" in generate_kwargs:
                                generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to(model_device)
                            
                            # Try generation with explicit BFloat16
                            generated_ids = self.model.generate(
                                input_features,
                                **generate_kwargs
                            )
                            if self.verbose:
                                print("XPU with BFloat16 failed. Falling back to CPU.")
                            self.model = self.model.to("cpu", dtype=torch.float32)
                            input_features = input_features.to("cpu", dtype=torch.float32)
                            
                            if "decoder_input_ids" in generate_kwargs:
                                generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to("cpu")
                            
                            # Generate on CPU
                            generated_ids = self.model.generate(
                                input_features,
                                **generate_kwargs
                            )
                            # End fallback to CPU with FP32 model
                        else:
                            # If model is not using BFloat16, fall back to CPU directly
                            if self.verbose:
                                print("Falling back to CPU processing.")
                            self.model = self.model.to("cpu", dtype=torch.float32)
                            input_features = input_features.to("cpu", dtype=torch.float32)
                            
                            if "decoder_input_ids" in generate_kwargs:
                                generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to("cpu")
                            
                            # Generate on CPU
                            generated_ids = self.model.generate(
                                input_features,
                                **generate_kwargs
                            )
                            # End fallback to CPU with FP32 model
                else:
                    # For other errors, re-raise
                    raise
