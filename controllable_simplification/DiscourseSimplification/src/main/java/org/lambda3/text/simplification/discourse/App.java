/*
 * ==========================License-Start=============================
 * DiscourseSimplification : App
 *
 * Copyright © 2017 Lambda³
 *
 * GNU General Public License 3
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 * ==========================License-End==============================
 */

package org.lambda3.text.simplification.discourse;

import org.lambda3.text.simplification.discourse.processing.DiscourseSimplifier;
import org.lambda3.text.simplification.discourse.processing.ProcessingType;
import org.lambda3.text.simplification.discourse.model.SimplificationContent;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import java.util.Scanner;
import java.util.ArrayList;
import java.io.FileNotFoundException;

public class App {
    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger(App.class);
    private static final DiscourseSimplifier DISCOURSE_SIMPLIFIER = new DiscourseSimplifier();

    private static void saveLines(File file, List<String> lines) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            bw.write(lines.stream().collect(Collectors.joining("\n")));

            // no need to close it.
            //bw.close()
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<String> get_sentences_list_from_file(File file){
        System.out.println("Getting list of sentences");
        List<String> sentences = new ArrayList<String>();
        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        while (fileScanner.hasNextLine()) {
            // Some additional code for storing the output to an array around here? :]
            String nextLine = fileScanner.nextLine();
            System.out.println(nextLine);
            sentences.add(nextLine);
        }

        System.out.print(sentences.size());
        System.out.println(sentences);
        return sentences;
    }
    public static void main(String[] args) throws IOException {
        SimplificationContent content = DISCOURSE_SIMPLIFIER.doDiscourseSimplification(get_sentences_list_from_file(new File("input.txt")), ProcessingType.SEPARATE);
        content.serializeToJSON(new File("output.json"));
        saveLines(new File("output_default.txt"), Arrays.asList(content.defaultFormat(false)));
        saveLines(new File("output_flat.txt"), Arrays.asList(content.flatFormat(false)));
        LOGGER.info("done");
    }
}
