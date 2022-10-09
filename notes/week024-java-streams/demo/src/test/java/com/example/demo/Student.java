package com.example.demo;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Student {
	private String name;
	private Integer age;
	private Long number;
	private String interests;
}
